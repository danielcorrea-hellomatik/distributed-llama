#include "nn/nn-core.hpp"
#include "nn/nn-config-builder.hpp"
#include "nn/nn-cpu.hpp"
#include "nn/nn-cpu-ops.hpp"
#include "nn/nn-network.hpp"
#include "nn/nn-executor.hpp"
#include "llm.hpp"
#include "tokenizer.hpp"
#include "app.hpp"
#include <stdexcept>
#include <cmath>

static void inference(AppInferenceContext *context) {
    if (context->args->prompt == nullptr)
        throw std::runtime_error("Prompt is required");
    if (context->args->steps == 0)
        throw std::runtime_error("Number of steps is required");

    std::vector<int> inputTokensVec(std::strlen(context->args->prompt) + 3);
    int *inputTokens = inputTokensVec.data();

    NnUint pos = 0;
    int nInputTokens;
    context->tokenizer->encode(context->args->prompt, inputTokens, &nInputTokens, true, true);

    if (nInputTokens > context->header->seqLen)
        throw std::runtime_error("The number of prompt tokens is greater than the sequence length");
    if (nInputTokens > context->args->steps)
        throw std::runtime_error("The number of prompt tokens is greater than the number of steps");

    NnSize sentBytes = 0;
    NnSize recvBytes = 0;
    NnUint evalTotalTime = 0;
    NnUint predTotalTime = 0;

    int token = inputTokens[pos];
    printf("%s\n", context->args->prompt);
    for (;;) {
        long remainingTokens = nInputTokens - 1 - (long)pos;
        if (remainingTokens <= 0)
            break;
        NnUint batchSize = remainingTokens < context->args->nBatches
            ? remainingTokens
            : context->args->nBatches;

        context->inference->setBatchSize(batchSize);
        context->inference->setPosition(pos);
        for (NnUint i = 0; i < batchSize; i++)
            context->inference->setToken(i, inputTokens[pos + i]);

        context->inference->forward();

        pos += batchSize;
        token = inputTokens[pos + 1];

        if (context->network != nullptr)
            context->network->getStats(&sentBytes, &recvBytes);

        NnUint evalTime = context->executor->getTotalTime(STEP_EXECUTE_OP);
        NnUint syncTime = context->executor->getTotalTime(STEP_SYNC_NODES);
        printf("🔷️ Eval%5u ms Sync%5u ms | Sent%6zu kB Recv%6zu kB | (%d tokens)\n",
            evalTime / 1000,
            syncTime / 1000,
            sentBytes / 1024,
            recvBytes / 1024,
            batchSize);
        evalTotalTime += evalTime + syncTime;
    }

    fflush(stdout);

    context->inference->setBatchSize(1);
    context->tokenizer->resetDecoder();

    const NnUint maxPos = std::min(context->header->seqLen, context->args->steps);
    for (; pos < maxPos; pos++) {
        context->inference->setPosition(pos);
        context->inference->setToken(0, token);
        context->inference->forward();

        token = context->sampler->sample(context->inference->logitsPipe);

        char *piece = context->tokenizer->decode(token);

        if (context->network != nullptr)
            context->network->getStats(&sentBytes, &recvBytes);

        NnUint predTime = context->executor->getTotalTime(STEP_EXECUTE_OP);
        NnUint syncTime = context->executor->getTotalTime(STEP_SYNC_NODES);
        printf("🔶 Pred%5u ms Sync%5u ms | Sent%6zu kB Recv%6zu kB | %s\n",
            predTime / 1000,
            syncTime / 1000,
            sentBytes / 1024,
            recvBytes / 1024,
            piece == nullptr ? "~" : piece);
        fflush(stdout);
        predTotalTime += predTime + syncTime;
    }

    NnUint nEvalTokens = nInputTokens - 1;
    NnUint nPredTokens = pos - nEvalTokens;
    float evalTotalTimeMs = evalTotalTime / 1000.0;
    float predTotalTimeMs = predTotalTime / 1000.0;
    printf("\n");
    printf("Evaluation\n");
    printf("   nBatches: %d\n", context->args->nBatches);
    printf("    nTokens: %d\n", nEvalTokens);
    printf("   tokens/s: %3.2f (%3.2f ms/tok)\n",
        (nEvalTokens * 1000) / evalTotalTimeMs,
        evalTotalTimeMs / ((float) nEvalTokens));
    printf("Prediction\n");
    printf("    nTokens: %d\n", nPredTokens);
    printf("   tokens/s: %3.2f (%3.2f ms/tok)\n",
        (nPredTokens * 1000) / predTotalTimeMs,
        predTotalTimeMs / ((float) nPredTokens));
}

// Lossless prompt-lookup speculative decoding (bit-exact under greedy).
// Each step drafts up to K tokens by matching the last specMin tokens against the
// existing sequence (PLD), then verifies ALL drafts in ONE batched forward. Because
// decode is memory-bandwidth-bound, that single weight read is amortized over K+1
// positions. Only tokens equal to the model's own argmax are emitted, so the output
// is byte-identical to plain greedy decode.
static void inferenceSpec(AppInferenceContext *context) {
    if (context->args->prompt == nullptr)
        throw std::runtime_error("Prompt is required");
    if (context->args->steps == 0)
        throw std::runtime_error("Number of steps is required");

    const NnUint vocab = context->header->vocabSize;
    const NnUint seqLen = context->header->seqLen;
    const NnUint nBatches = context->args->nBatches;
    NnUint K = context->args->specNgram;                 // max draft tokens / step
    if (K + 1 > nBatches) K = nBatches - 1;              // verify batch (K+1) must fit
    const NnUint specMin = context->args->specMin < 1 ? 1 : context->args->specMin;

    std::vector<int> inputTokensVec(std::strlen(context->args->prompt) + 3);
    int *inputTokens = inputTokensVec.data();
    int nInputTokens;
    context->tokenizer->encode(context->args->prompt, inputTokens, &nInputTokens, true, true);

    if (nInputTokens > (int)seqLen)
        throw std::runtime_error("The number of prompt tokens is greater than the sequence length");
    if (nInputTokens > (int)context->args->steps)
        throw std::runtime_error("The number of prompt tokens is greater than the number of steps");

    std::vector<int> seq(inputTokens, inputTokens + nInputTokens); // full history (KV truth + PLD corpus)

    NnUint predTotalTime = 0;
    NnUint pos = 0;
    printf("%s\n", context->args->prompt);

    // ---- prefill (identical to plain inference) ----
    for (;;) {
        long remainingTokens = nInputTokens - 1 - (long)pos;
        if (remainingTokens <= 0) break;
        NnUint batchSize = remainingTokens < (long)nBatches ? (NnUint)remainingTokens : nBatches;
        context->inference->setBatchSize(batchSize);
        context->inference->setPosition(pos);
        for (NnUint i = 0; i < batchSize; i++)
            context->inference->setToken(i, inputTokens[pos + i]);
        context->inference->forward();
        pos += batchSize;
    }
    fflush(stdout);
    context->tokenizer->resetDecoder();

    // ---- speculative decode loop ----
    const NnUint maxPos = std::min(seqLen, context->args->steps);
    pos = nInputTokens - 1;                 // position of the current (unforwarded) token
    int token = inputTokens[nInputTokens - 1];

    NnUint nPredTokens = 0, nSpecSteps = 0, nDraftTokens = 0, nAcceptedDraft = 0;
    NnUint kCur = (K > 0) ? 1u : 0u;   // adaptive draft cap: grows on accept, shrinks on miss

    while (pos < maxPos) {
        // adaptive draft length: 0 when we keep missing (batch=1, no compute penalty),
        // with a periodic probe so we re-detect predictable/repetitive stretches.
        NnUint kAllow = kCur;
        if (kAllow == 0 && K > 0 && (nSpecSteps % 8u) == 0u) kAllow = 1;

        // --- draft via prompt-lookup: most-recent prior match of the last specMin tokens ---
        std::vector<int> draft;
        NnUint nseq = (NnUint)seq.size();
        if (kAllow > 0 && nseq >= specMin) {
            for (long start = (long)nseq - (long)specMin - 1; start >= 0; start--) {
                bool match = true;
                for (NnUint j = 0; j < specMin; j++)
                    if (seq[start + j] != seq[nseq - specMin + j]) { match = false; break; }
                if (match) {
                    NnUint from = (NnUint)start + specMin;
                    for (NnUint d = 0; d < kAllow && from + d < nseq; d++)
                        draft.push_back(seq[from + d]);
                    break;
                }
            }
        }
        NnUint kd = (NnUint)draft.size();
        while (kd > 0 && pos + kd >= maxPos) { draft.pop_back(); kd--; }   // keep positions in range
        NnUint B = 1 + kd;

        // --- one batched forward verifies token + all drafts ---
        context->inference->setBatchSize(B);
        context->inference->setPosition(pos);
        context->inference->setToken(0, token);
        for (NnUint i = 0; i < kd; i++)
            context->inference->setToken(i + 1, draft[i]);
        context->inference->forward();
        nSpecSteps++;
        nDraftTokens += kd;

        // a[0] = true next token (always accept)
        int a0 = context->sampler->sample(context->inference->logitsPipe);
        seq.push_back(a0);
        { char *piece = context->tokenizer->decode(a0); if (piece) printf("%s", piece); }
        nPredTokens++; pos++; token = a0;

        // accept drafts while they match the model's argmax at each verified position
        NnUint accThis = 0;
        for (NnUint i = 1; i <= kd && pos < maxPos; i++) {
            if (draft[i - 1] != token) break;             // draft[i-1] guessed position pos -> true is `token`
            int ai = context->sampler->sample(context->inference->logitsPipe + (size_t)i * vocab);
            seq.push_back(ai);
            { char *piece = context->tokenizer->decode(ai); if (piece) printf("%s", piece); }
            nPredTokens++; nAcceptedDraft++; accThis++; pos++; token = ai;
        }
        // adapt: grow fast when all drafts hit, shrink when none hit (-> 0 = no penalty)
        if (kd > 0) {
            if (accThis == kd)      kCur = (kCur + 2 < K) ? kCur + 2 : K;
            else if (accThis == 0)  kCur = (kCur > 0) ? kCur - 1 : 0;
        }
        fflush(stdout);
        predTotalTime += context->executor->getTotalTime(STEP_EXECUTE_OP) + context->executor->getTotalTime(STEP_SYNC_NODES);
    }

    float predTotalTimeMs = predTotalTime / 1000.0f;
    printf("\n\n");
    printf("Prediction (speculative ngram=%u min=%u)\n", context->args->specNgram, specMin);
    printf("    nTokens: %u\n", nPredTokens);
    printf("     nSteps: %u\n", nSpecSteps);
    printf("     accept: %u/%u drafts  (avg %.2f tok/step)\n",
        nAcceptedDraft, nDraftTokens, nSpecSteps ? (float)nPredTokens / (float)nSpecSteps : 0.0f);
    printf("   tokens/s: %3.2f (%3.2f ms/tok)\n",
        predTotalTimeMs > 0 ? (nPredTokens * 1000) / predTotalTimeMs : 0.0f,
        nPredTokens ? predTotalTimeMs / (float)nPredTokens : 0.0f);
}

static NnUint readStdin(const char *guide, char *buffer, NnUint size) {
    std::fflush(stdin);
    std::printf("%s", guide);
    if (std::fgets(buffer, size, stdin) != NULL) {
        NnUint length = std::strlen(buffer);
        if (length > 0 && buffer[length - 1] == '\n') {
            buffer[length - 1] = '\0';
            length--;
        }
        return length;
    }
    return 0;
}

static void perplexity(AppInferenceContext *context) {
    if (context->args->prompt == nullptr)
        throw std::runtime_error("Prompt is required");

    std::vector<int> inputTokensVec(std::strlen(context->args->prompt) + 3);
    int *inputTokens = inputTokensVec.data();

    int nInputTokens;
    context->tokenizer->encode(context->args->prompt, inputTokens, &nInputTokens, true, true);

    printf("Evaluating %d tokens...\n", nInputTokens);

    float totalLogProb = 0.0f;
    NnUint pos = 0;

    context->inference->setBatchSize(1);

    for (pos = 0; pos < nInputTokens - 1; pos++) {
        context->inference->setPosition(pos);
        context->inference->setToken(0, inputTokens[pos]);
        context->inference->forward();

        float *logits = context->inference->logitsPipe;
        softmax_F32(logits, context->header->vocabSize);

        int targetToken = inputTokens[pos + 1];
        float prob = logits[targetToken];

        totalLogProb += std::log(std::max(prob, 1e-30f));
        printf("%5d / %d, prob=%f\n", pos + 1, nInputTokens - 1, prob);
    }

    float avgLogProb = totalLogProb / (float)(nInputTokens - 1);
    float perplexity = expf(-avgLogProb);

    printf("\n");
    printf("Results\n");
    printf("   perplexity: %f (lower = better)\n", perplexity);
    printf("   avgLogProb: %f\n", avgLogProb);
    printf("   bitPerToken: %f\n", -avgLogProb / std::log(2.0));
}

static void chat(AppInferenceContext *context) {
    const NnUint seqLen = context->header->seqLen;
    char prompt[2048];

    TokenizerChatStops stops(context->tokenizer);
    ChatTemplateGenerator templateGenerator(context->args->chatTemplateType, context->tokenizer->chatTemplate, stops.stops[0]);
    EosDetector eosDetector(stops.nStops, context->tokenizer->eosTokenIds.data(), stops.stops, stops.maxStopLength, stops.maxStopLength);

    const NnUint sysPromptLength = readStdin("💻 System prompt (optional): ", prompt, sizeof(prompt));
    std::vector<ChatItem> deltaItems;
    if (sysPromptLength > 0)
        deltaItems.push_back(ChatItem{"system", prompt});

    NnUint pos = 0;
    NnUint userPromptLength;
    int token;
    int nInputTokens;
    do {
        do {
            userPromptLength = readStdin("\n👱 User\n> ", prompt, sizeof(prompt));
        } while (userPromptLength == 0);

        deltaItems.push_back(ChatItem{"user", prompt});

        GeneratedChat inputPrompt = templateGenerator.generate(deltaItems.size(), deltaItems.data(), true);
        std::unique_ptr<int[]> inputTokensPtr(new int[inputPrompt.length + 2]);
        int *inputTokens = inputTokensPtr.get();

        bool isStart = pos == 0;
        context->tokenizer->encode((char*)inputPrompt.content, inputTokens, &nInputTokens, isStart, true);

        NnUint userPromptEndPos = (NnUint)std::min<unsigned int>(seqLen, pos + nInputTokens - 1);
        for (NnUint i = 0; ;) {
            int remainingTokens = userPromptEndPos - pos;
            if (remainingTokens <= 0)
                break;
            NnUint batchSize = remainingTokens < context->args->nBatches
                ? remainingTokens
                : context->args->nBatches;

            context->inference->setBatchSize(batchSize);
            context->inference->setPosition(pos);
            for (NnUint j = 0; j < batchSize; j++)
                context->inference->setToken(j, inputTokens[i + j]);

            context->inference->forward();

            i += batchSize;
            pos += batchSize;
            token = inputTokens[i + 1];
        }

        context->inference->setBatchSize(1);
        context->tokenizer->resetDecoder();

        printf("\n🤖 Assistant\n");
        if (inputPrompt.publicPrompt != nullptr)
            printf("%s", inputPrompt.publicPrompt);

        while (pos < seqLen) {
            context->inference->setPosition(pos);
            context->inference->setToken(0, token);
            context->inference->forward();

            token = context->sampler->sample(context->inference->logitsPipe);

            char *piece = context->tokenizer->decode(token);
            EosDetectorType eosType = eosDetector.append(token, piece);
            if (eosType == NOT_EOS || eosType == EOS) {
                char *delta = eosDetector.getDelta();
                if (delta != nullptr) {
                    printf("%s", delta);
                    fflush(stdout);
                }
                eosDetector.reset();
            }
            pos++;
            if (eosType == EOS) break;
        }

        deltaItems.clear();
    } while (pos < seqLen);

    printf("(end of context)\n");
}

int main(int argc, char **argv) {
    initQuants();
    initSockets();

    int returnCode = EXIT_SUCCESS;
    try {
        AppCliArgs args = AppCliArgs::parse(argc, argv, true);
        if (std::strcmp(args.mode, "inference") == 0) {
            args.benchmark = true;
            if (args.specNgram > 0)
                runInferenceApp(&args, &inferenceSpec);
            else
                runInferenceApp(&args, &inference);
        } else if (std::strcmp(args.mode, "perplexity") == 0)
            runInferenceApp(&args, &perplexity);
        else if (std::strcmp(args.mode, "chat") == 0)
            runInferenceApp(&args, &chat);
        else if (std::strcmp(args.mode, "worker") == 0)
            runWorkerApp(&args);
        else
            throw std::runtime_error("Unsupported mode");
    } catch (const std::exception &e) {
        printf("🚨 Critical error: %s\n", e.what());
        returnCode = EXIT_FAILURE;
    }

    cleanupSockets();
    return returnCode;
}

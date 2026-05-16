#include <cstdlib>
#include <stdexcept>
#include <cassert>
#include <cstring>
#include "nn-executor.hpp"

void NnFakeNodeSynchronizer::sync(NnUint segmentIndex, NnUint nThreads, NnUint threadIndex) {
    // Nothing
}


static inline NnByte *alignedAlloc(size_t size) {
    void *ptr = nullptr;
    size_t aligned = (size + 63) & ~63;  // round up to 64
    if (posix_memalign(&ptr, 64, aligned) != 0 || ptr == nullptr)
        throw std::runtime_error("aligned alloc failed");
    return (NnByte *)ptr;
}

static inline void alignedFree(NnByte *p) {
    free(p);
}

NnNetExecution::NnNetExecution(NnUint nThreads, NnNetConfig *netConfig) {
    this->nThreads = nThreads;
    this->nBatches = netConfig->nBatches;
    this->nPipes = netConfig->nPipes;
    this->batchSize = 0; // This value must be overwritten before calling forward

    pipes = new NnByte *[netConfig->nPipes];
    for (NnUint pipeIndex = 0; pipeIndex < netConfig->nPipes; pipeIndex++) {
        NnPipeConfig *pipeConfig = &netConfig->pipes[pipeIndex];
        NnByte *pipe = alignedAlloc(pipeConfig->size.nBytes);
        std::memset(pipe, 0, pipeConfig->size.nBytes);
        pipes[pipeIndex] = pipe;
    }
}

NnNetExecution::~NnNetExecution() {
    for (NnUint pipeIndex = 0; pipeIndex < nPipes; pipeIndex++)
        delete[] pipes[pipeIndex];
    delete[] pipes;
}

void NnNetExecution::setBatchSize(NnUint batchSize) {
    assert(batchSize <= nBatches);
    this->batchSize = batchSize;
}

NnExecutorDevice::NnExecutorDevice(NnDevice *device, int segmentFrom, int segmentTo) {
    this->device = std::unique_ptr<NnDevice>(device);
    this->segmentFrom = segmentFrom;
    this->segmentTo = segmentTo;
}

NnExecutorException::NnExecutorException(const std::string message)
    : std::runtime_error(message)
{}

// Forward declaration: defined later in this file.
static inline void *executorWorkerLoop(void *arg);

NnExecutor::NnExecutor(NnNetConfig *netConfig, NnNodeConfig *nodeConfig, std::vector<NnExecutorDevice> *devices, NnNetExecution *netExecution, NnNodeSynchronizer *synchronizer, bool benchmark)
    : segments(nodeConfig->nSegments), steps()
{
    NnUint maxNThreads = 0;
    for (NnExecutorDevice &d : *devices) {
        if (d.device->maxNThreads() > maxNThreads)
            maxNThreads = d.device->maxNThreads();
    }
    if (netExecution->nThreads > maxNThreads)
        throw std::invalid_argument("This configuration supports max " + std::to_string(maxNThreads) + " threads");

    this->netExecution = netExecution;
    this->nodeConfig = nodeConfig;

    bool useSynchronizer = netConfig->nNodes > 1;
    for (NnUint segmentIndex = 0; segmentIndex < nodeConfig->nSegments; segmentIndex++) {
        NnDevice *device = nullptr;
        for (NnExecutorDevice &d : *devices) {
            if (
                (d.segmentFrom == -1 && d.segmentTo == -1) ||
                (segmentIndex >= d.segmentFrom && segmentIndex <= d.segmentTo)
            ) {
                device = d.device.get();
                break;
            }
        }
        if (device == nullptr)
            throw std::invalid_argument("Cannot locate device for segment " + std::to_string(segmentIndex));

        NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];
        if (segmentConfig->nOps > 0) {
            NnDeviceSegment *segment = device->createSegment(segmentIndex);
            segments[segmentIndex] = std::unique_ptr<NnDeviceSegment>(segment);

            for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++)
                steps.push_back(NnExecutorStep{ STEP_EXECUTE_OP, segment, opIndex, &segmentConfig->ops[opIndex] });
        }
        if (useSynchronizer && segmentConfig->nSyncs > 0)
            steps.push_back(NnExecutorStep{ STEP_SYNC_NODES, nullptr, segmentIndex, nullptr });
    }

    steps.shrink_to_fit();

    context.nThreads = netExecution->nThreads;
    context.synchronizer = synchronizer;
    context.nSteps = (NnUint)steps.size();
    context.steps = steps.data();
    if (benchmark)
        context.timer = new Timer();
    else
        context.timer = nullptr;

    threads = new NnExecutorThread[netExecution->nThreads];
    for (NnUint threadIndex = 0; threadIndex < netExecution->nThreads; threadIndex++) {
        NnExecutorThread *thread = &threads[threadIndex];
        thread->threadIndex = threadIndex;
        thread->context = &context;
    }

    // Persistent worker pool: spawn N-1 threads once and reuse across every
    // forward() call. Thread 0 is the calling (main) thread and is not
    // created here.
    context.generation.store(0, std::memory_order_relaxed);
    context.workersDone.store(0, std::memory_order_relaxed);
    context.shutdown.store(false, std::memory_order_relaxed);
    for (NnUint threadIndex = 1; threadIndex < netExecution->nThreads; threadIndex++) {
        int result = pthread_create(&threads[threadIndex].handler, NULL, (PthreadFunc)executorWorkerLoop, (void *)&threads[threadIndex]);
        assert(result == 0 && "Failed to create worker thread");
    }
}

NnExecutor::~NnExecutor() {
    // Signal worker pool to exit and join.
    context.shutdown.store(true, std::memory_order_release);
    context.generation.fetch_add(1, std::memory_order_release);
    for (NnUint threadIndex = 1; threadIndex < netExecution->nThreads; threadIndex++)
        pthread_join(threads[threadIndex].handler, NULL);

    if (context.timer != nullptr)
        delete context.timer;
    delete[] threads;
}

void NnExecutor::loadWeight(const char *name, NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) {
    for (NnUint segmentIndex = 0; segmentIndex < nodeConfig->nSegments; segmentIndex++) {
        NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];
        for (NnUint i = 0; i < segmentConfig->nOps; i++) {
            NnOpConfig *opConfig = &segmentConfig->ops[i];
            if (opConfig->index == opIndex && std::strcmp(opConfig->name, name) == 0) {
                NnDeviceSegment *segment = segments[segmentIndex].get();
                assert(segment != nullptr);
                segment->loadWeight(i, offset, nBytes, weight);
                return;
            }
        }
    }
    throw std::invalid_argument("Cannot locate op by name: " + std::string(name));
}

inline void executeStep(NnExecutorStep *step, NnUint nThreads, NnExecutorThread *thread, NnExecutorContext *context) {
    if (step->type == STEP_EXECUTE_OP) {
        step->segment->forward(step->arg0, nThreads, thread->threadIndex, context->batchSize);
    } else if (step->type == STEP_SYNC_NODES) {
        context->synchronizer->sync(step->arg0, nThreads, thread->threadIndex);
    } else {
        throw std::invalid_argument("Unsupported step type");
    }
}

static inline void executorStepLoop(NnExecutorThread *thread) {
    NnExecutorContext *context = thread->context;
    NnUint nThreads = context->nThreads;
    NnUint doneCount = nThreads - 1;

    while (context->isAlive.load(std::memory_order_acquire)) {
        const unsigned int currentStepIndex = context->currentStepIndex.load(std::memory_order_acquire);
        if (currentStepIndex == context->nSteps)
            break;

        NnExecutorStep *step = &context->steps[currentStepIndex];
        try {
            executeStep(step, nThreads, thread, context);
        } catch (const std::runtime_error &e) {
            context->isAlive.store(false, std::memory_order_release);
            printf("🚨 Execution error: %s\n", e.what());
            break;
        }

        NnUint currentCount = context->doneThreadCount.fetch_add(1, std::memory_order_acq_rel);
        if (currentCount == doneCount) {
            if (context->timer != nullptr) {
                NnUint time = context->timer->elapsedMicroseconds();
                context->totalTime[step->type] += time;
                context->timer->reset();
            }

            context->doneThreadCount.store(0, std::memory_order_relaxed);
            context->currentStepIndex.fetch_add(1, std::memory_order_release);
        } else {
            while (
                context->currentStepIndex.load(std::memory_order_acquire) == currentStepIndex &&
                context->isAlive.load(std::memory_order_acquire)
            );
        }
    }
}

// Worker thread main loop: waits for a new forward() call (generation bump),
// runs the step loop, signals done, repeats. Lifetime spans the whole
// NnExecutor; saves one pthread_create + pthread_join per token.
static inline void *executorWorkerLoop(void *arg) {
    NnExecutorThread *thread = (NnExecutorThread *)arg;
    NnExecutorContext *context = thread->context;
    unsigned long long lastGen = 0;

    while (true) {
        // Wait for next forward() call. Spin on the generation counter; pause
        // hint reduces power and avoids hammering the cache line.
        unsigned long long gen;
        for (;;) {
            gen = context->generation.load(std::memory_order_acquire);
            if (gen != lastGen) break;
            if (context->shutdown.load(std::memory_order_acquire)) return nullptr;
#if defined(__aarch64__) || defined(__arm__)
            asm volatile ("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
            asm volatile ("pause" ::: "memory");
#endif
        }
        if (context->shutdown.load(std::memory_order_acquire)) return nullptr;
        lastGen = gen;

        executorStepLoop(thread);

        context->workersDone.fetch_add(1, std::memory_order_release);
    }
}

void NnExecutor::forward() {
    assert(netExecution->batchSize > 0);

    NnUint nThreads = netExecution->nThreads;
    context.isAlive.store(true, std::memory_order_relaxed);
    context.currentStepIndex.store(0, std::memory_order_relaxed);
    context.doneThreadCount.store(0, std::memory_order_relaxed);
    context.workersDone.store(0, std::memory_order_relaxed);
    context.batchSize = netExecution->batchSize;

    if (context.timer != nullptr) {
        std::memset(context.totalTime, 0, sizeof(context.totalTime));
        context.timer->reset();
    }

    // Release the workers: they have been spinning on this counter inside
    // executorWorkerLoop and will pick up the new generation and start work.
    context.generation.fetch_add(1, std::memory_order_release);

    // Run thread 0 inline on the calling thread.
    executorStepLoop(&threads[0]);

    // Wait for the workers to finish this generation.
    const NnUint expectedDone = nThreads - 1;
    while (context.workersDone.load(std::memory_order_acquire) < expectedDone) {
#if defined(__aarch64__) || defined(__arm__)
        asm volatile ("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
        asm volatile ("pause" ::: "memory");
#endif
    }

    if (!context.isAlive.load(std::memory_order_acquire))
        throw NnExecutorException("Execution failed in one of the threads");
}

NnUint NnExecutor::getTotalTime(NnExecutorStepType type) {
    assert((NnUint)type < N_STEP_TYPES);
    return context.totalTime[type];
}

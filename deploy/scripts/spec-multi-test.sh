#!/usr/bin/env bash
set -e

TESTS=(
  "code:def fibonacci(n):
    if n <= 1: return n
    return fibonacci(n-1) + fibonacci(n-2)
fibonacci(10) = "

  "json:{\"users\": [{\"id\": 1, \"name\": \"Alice\"}, {\"id\": 2, \"name\": \"Bob\"}], \"total\": "

  "list:Items in the list are:
1. apples
2. bananas
3. cherries
4. dates
5. elderberries
6. "

  "reason:Given that all humans are mortal, and Socrates is human, therefore Socrates is mortal. This is an example of a valid deductive argument where "

  "dialog:User: Hello, how are you?
Assistant: I'm doing well, thank you for asking. How can I help you today?
User: I'd like to know more about"

  "repeat:The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. "
)

MODELS=( "DeepSeek-8B" "Llama-3.1-8B" )
MODEL_PATHS=(
  "/home/rpi/distributed-llama/models/DeepSeek-8B.gguf"
  "/home/rpi/distributed-llama/models/Llama-3.1-8B.gguf"
)

RESULTS_FILE="/tmp/spec_results_$(date +%s).txt"

echo "=== Multi-Prompt Speculative Decoding Test ===" | tee "$RESULTS_FILE"
echo "Date: $(date)" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

for ((m=0; m<${#MODELS[@]}; m++)); do
  MODEL="${MODELS[$m]}"
  MODEL_PATH="${MODEL_PATHS[$m]}"
  
  [[ ! -f "$MODEL_PATH" ]] && { echo "⚠ Model $MODEL not found, skipping"; continue; }
  
  echo "━━━ Model: $MODEL ━━━" | tee -a "$RESULTS_FILE"
  
  for test_spec in "${TESTS[@]}"; do
    IFS=':' read -r test_name test_prompt <<< "$test_spec"
    
    echo "  Test: $test_name" | tee -a "$RESULTS_FILE"
    
    # Run greedy (reference)
    greedy_out=$(ssh -o BatchMode=yes -o ConnectTimeout=10 rpi@rpi-1005 bash -c "
cd ~/dllama-spec
echo '$test_prompt' | timeout 20 ./dllama \\
  --model '$MODEL_PATH' --nthreads 3 --chat --spec-ngram 0 2>/dev/null | tail -20
" 2>/dev/null || echo "")
    
    # Run speculative
    spec_out=$(ssh -o BatchMode=yes -o ConnectTimeout=10 rpi@rpi-1005 bash -c "
cd ~/dllama-spec
echo '$test_prompt' | timeout 20 ./dllama \\
  --model '$MODEL_PATH' --nthreads 3 --chat --spec-ngram 3 2>/dev/null | tail -20
" 2>/dev/null || echo "")
    
    # Extract metrics (safely)
    greedy_tokens=$(echo "$greedy_out" | grep -oP 'nTokens: \K\d+' | head -1 || echo "—")
    spec_tokens=$(echo "$spec_out" | grep -oP 'nTokens: \K\d+' | head -1 || echo "—")
    spec_accept=$(echo "$spec_out" | grep -oP 'accept: \K[^(]+' | head -1 || echo "—")
    
    greedy_tps=$(echo "$greedy_out" | grep -oP 'tokens/s: \K[0-9.]+' | head -1 || echo "—")
    spec_tps=$(echo "$spec_out" | grep -oP 'tokens/s: \K[0-9.]+' | head -1 || echo "—")
    
    # Calculate speedup
    if [[ "$greedy_tps" != "—" && "$spec_tps" != "—" ]]; then
      speedup=$(echo "scale=2; $spec_tps / $greedy_tps" | bc)
    else
      speedup="—"
    fi
    
    printf "    %-12s greedy=%s tok/s | spec=%s tok/s (speedup=%.2fx, accept=%s)\n" \
      "" "$greedy_tps" "$spec_tps" "$speedup" "$spec_accept" | tee -a "$RESULTS_FILE"
  done
  
  echo "" >> "$RESULTS_FILE"
done

echo "=== Results saved to $RESULTS_FILE ===" | tee -a "$RESULTS_FILE"
cat "$RESULTS_FILE"

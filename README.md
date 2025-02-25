# advanced_rag


### vLLM:

- vLLM uses page attention to imporve LLM speed and scale.
- There are some tradeoffs to using vLLM compared to say Huggingface pipeline.
-   1. Cold boot: takes longer to cold bootup using vLLM
    2. Streaming words: You may notice the app merges words, esoecially at the beinging of the sentence, Im not alone with this issue. vLLM grammar is perfect for when the text is outputted at once. If try to stream it to teh user for better UI we get spelling issues (cintrolling for which can make the code quiete complex). 

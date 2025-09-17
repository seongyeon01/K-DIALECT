# K-DIALECT : KOREAN DIALECT-AWARE FACE-BASED SPEECH SYNTHESIS
Our demo page is released in https://dxlabskku.github.io/K-Dialect/

## Abstact
Face-based speech synthesis emerged as a promising alternative in settings where clean reference audio data are scarce or noisy. However, existing studies hardly addressed the challenge of modeling dialectal prosody, which is essential for preserving linguistic diversity and naturalness. To address this issue, we propose K-DIALECT, a multimodal text-to-speech(TTS) framework that generates speech from only a face image and dialect identifier. The model incorporates a face encoder to disentangle the speaker identity and style, a dialect-conditioned pitch predictor to explicitly model prosody, and a FiLM-based fusion module to integrate the pitch and style into a unified representation. Experiments on six Korean dialects show that our method achieves higher dialectal fluency and naturalness than a strong voice-based baseline while maintaining competitive speaker similarity. K-DIALECT achieved a SECS of 0.70 and a WER of 0.25, confirming that it preserves both similarity and intelligibility. Subjective evaluations further reveal that the improvements are especially pronounced for dialects with distinctive pitch variations, and listeners consistently preferred our outputs. This work represents the first attempt to extend face-based TTS to low-resource dialectal synthesis, providing a foundation for future research on multilingual and extremely low-resource speech synthesis.
## Train
```
python train.py
```

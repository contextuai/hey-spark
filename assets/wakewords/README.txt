Wake-word model files for Hey Spark
===================================

hey_spark.onnx (openWakeWord) is bundled. Picovoice .ppn keyword files are licensed per account,
so they are not committed: put yours here and point config.yaml at them. To train your own wake
word, see docs/wakeword.md. Paths are relative to the folder holding config.yaml.

Porcupine (optional, wakeword.backend: porcupine)
  File:   hey_spark_windows.ppn
  Config: wakeword.options.porcupine.keyword_path: assets/wakewords/hey_spark_windows.ppn
  How:    Picovoice Console (https://console.picovoice.ai/) > Porcupine > phrase "Hey Spark",
          platform "Windows (x86_64)". Download the .zip and copy the .ppn here, renamed to
          hey_spark_<platform>.ppn (for example hey_spark_windows.ppn). A .ppn only works on the
          platform it was trained for and with the same Porcupine major version.
  Key:    PICOVOICE_ACCESS_KEY in .env (never in config.yaml).

openWakeWord (current backend, wakeword.backend: openwakeword)
  File:   hey_spark.onnx ("Hey Spark", self-contained). Trained with
          tools/wakeword_trainer.ipynb in Colab; docs/wakeword.md shows how to train your own.
          If a PyTorch export produces a separate <name>.onnx.data weights file, run the
          notebook's Step 12 to fold it into one file.
  License: CC BY-NC-SA 4.0 (non-commercial), see LICENSE in this folder.
  Config: wakeword.options.openwakeword.model_path: assets/wakewords/hey_spark.onnx
  Tuning: wakeword.options.openwakeword.threshold (0.6); lower = more wakes, higher = fewer.
          On synthetic voices "hey spark" scored 0.79-0.86, while "hey park", "hey sport",
          "hey spike" and "hey sparky" reached ~0.8 for some voices; everyday speech stayed ~0.
  Set model_path: null to fall back to the bundled "hey_jarvis" model (say "Hey Jarvis").

Frame sizes: Porcupine consumes 512-sample frames, openWakeWord 1280-sample frames, both at
16 kHz mono int16. spark/audio/stream.py re-chunks the mic stream for each.

Check the setup with: uv run python -m spark doctor

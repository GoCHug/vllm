hcr.meta-wulan01.hw-wulan.local/antsys/vllm:v0.23.0-openeuler-20260819150428

hcr.meta-guian02.guian.hw-a3.local/antsys/vllm:v0.23.0-a3-openeuler-20260818163431_aarch64

model-cli pull hcr.meta-guian02.guian.hw-a3.local/aistudio/modelhub_74000048_meta-llama-3-8b:148700128_20260921221233

hcr.meta-guian02.guian.hw-a3.local/aistudio/modelhub_97542_deepseek-v2-lite:36500041_20260318110950

hcr.meta-guian02.guian.hw-a3.local/aistudio/modelhub_41700011_qwen3.5-4b:112600047_20260303111458

itask create --name gggtest1 \
--image hcr.meta-guian02.guian.hw-a3.local/antsys/vllm:v0.23.0-a3-openeuler-20260818163431_aarch64 \
--model hcr.meta-guian02.guian.hw-a3.local/aistudio/modelhub_74000048_meta-llama-3-8b:148700128_20260921221233 \
--4card \
-t a3

itask create --name gggtest2 \
--image hcr.meta-guian02.guian.hw-a3.local/antsys/vllm:v0.23.0-a3-openeuler-20260818163431_aarch64 \
--model hcr.meta-guian02.guian.hw-a3.local/aistudio/modelhub_97542_deepseek-v2-lite:36500041_20260318110950 \
--4card \
-t a3

itask create --name gggtest3 \
--image hcr.meta-guian02.guian.hw-a3.local/antsys/vllm:v0.23.0-a3-openeuler-20260818163431_aarch64 \
--model hcr.meta-guian02.guian.hw-a3.local/aistudio/modelhub_41700011_qwen3.5-4b:112600047_20260303111458 \
--4card \
-t a3


itask create --name gggtest4 \
--image hcr.meta-guian02.guian.hw-a3.local/antsys/vllm:v0.23.0-a3-openeuler-20260818163431_aarch64 \
--model hcr.meta-guian02.guian.hw-a3.local/aistudio/modelhub_41900013_qwen3.5-0.8b:112800048_20260304125902 \
--4card \
-t a3
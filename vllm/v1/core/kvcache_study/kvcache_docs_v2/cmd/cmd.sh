hcr.meta-guian02.guian.hw-a3.local/antsys/vllm:v0.23.0-a3-openeuler-20260818163431_aarch64

model-cli pull hcr.meta-guian02.guian.hw-a3.local/aistudio/modelhub_74000048_meta-llama-3-8b:148700128_20260921221233


itask create --name gggtest \
--image hcr.meta-guian02.guian.hw-a3.local/antsys/vllm:v0.23.0-a3-openeuler-20260818163431_aarch64 \
--model hcr.meta-guian02.guian.hw-a3.local/aistudio/modelhub_74000048_meta-llama-3-8b:148700128_20260921221233 \
--4card \
-t a3

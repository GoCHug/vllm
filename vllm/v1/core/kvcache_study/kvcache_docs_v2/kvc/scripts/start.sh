vllm serve /home/admin/model-csi/models/modelhub_74000048_meta-llama-3-8b-148700128_20260921221233/model \
    --enforce-eager \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 2 > ./llama.log 2>&1 &

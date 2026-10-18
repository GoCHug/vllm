import json, urllib.request, time

PROMPT = ('KV cache 分页管理与前缀复用是推理引擎显存效率的两大支柱，'
          '多级块账本与链式哈希共同决定了缓存命中行为。' * 25)          + '请用不超过30个字总结上文主旨。'

def ask(tag):
    payload = json.dumps({'model': 'deepseek-v2-lite',
                          'messages': [{'role': 'user', 'content': PROMPT}],
                          'max_tokens': 48}).encode()
    req = urllib.request.Request('http://localhost:8100/v1/chat/completions',
                                 data=payload,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req) as r:
        d = json.load(r)
    dt = time.time() - t0
    print(f'[run {tag}] wall={dt:.2f}s finish={d["choices"][0]["finish_reason"]}')
    print(f'[run {tag}] usage={json.dumps(d["usage"])}')

ask('cold1')   # 第一次：冷缓存，完整 prefill
ask('warm2')   # 第二次：相同 prompt，期望命中前缀缓存
ask('warm3')   # 第三次：确认稳定命中
import json, urllib.request, time
PROMPT = ('KV cache 分页管理与前缀复用是推理引擎显存效率的两大支柱，'
          '多级块账本与链式哈希共同决定了缓存命中行为。' * 25)          + '请详细介绍缓存管理的三个关键阶段，每个阶段不少于150字。'
payload = json.dumps({'model': 'deepseek-v2-lite',
                      'messages': [{'role': 'user', 'content': PROMPT}],
                      'max_tokens': 512}).encode()
req = urllib.request.Request('http://localhost:8100/v1/chat/completions',
                             data=payload,
                             headers={'Content-Type': 'application/json'})
t0 = time.time()
with urllib.request.urlopen(req) as r:
    d = json.load(r)
print(f'wall={time.time()-t0:.1f}s finish={d["choices"][0]["finish_reason"]}')
print('usage=', json.dumps(d['usage']))

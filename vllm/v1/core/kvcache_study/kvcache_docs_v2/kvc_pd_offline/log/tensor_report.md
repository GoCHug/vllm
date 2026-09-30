# tensor_report（kvc_pd_offline 五级检查）
- verdict: **PASS**
- tensors: ['kv_D_1_aa9a175b.pt', 'kv_D_2_87786813.pt', 'kv_P_1_bcd07435.pt', 'kv_P_2_81a351d5.pt']
- L1 互证: [{'key': 'kv_D_1', 'log': 'kvc_D_reqp.log', 'n_checked': 320, 'n_equal': 320}, {'key': 'kv_D_2', 'log': 'kvc_D_reqr.log', 'n_checked': 448, 'n_equal': 448}, {'key': 'kv_P_1', 'log': 'kvc_P_reqp.log', 'n_checked': 320, 'n_equal': 320}, {'key': 'kv_P_2', 'log': 'kvc_P_reqr.log', 'n_checked': 384, 'n_equal': 384}]
- L2 Tx: [{'seq': 1, 'p_tok': 324, 'pairs_checked': 64, 'pairs_equal': 64}, {'seq': 2, 'p_tok': 486, 'pairs_checked': 64, 'pairs_equal': 64}]
- L3 明细条数: 128  L4: {'seq2_decode': {'rows': 34, 'health': '无', 'first3': [-5.375, 1.164, -1.539]}}
- issues: []

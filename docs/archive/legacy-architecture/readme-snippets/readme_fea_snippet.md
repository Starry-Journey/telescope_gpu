## FEA 案例与机械-光学灵敏度（v1）

`data/fea_cases/` 是根据两版设计资料生成的**合成参考工况**（不是已求解的 2.5 m FEA 模型）：

```bash
python3 -m products.sensitivity.generate_cases
python3 -m unittest discover -s web/tests -v
```

接口：

```text
GET /api/fea/cases
GET /api/fea/cases/{id}
GET /api/fea/sensitivity
```

页面“④ FEA 案例”显示面形热力图、15 项 Zernike、环境 S（重力/温度）。
真实 FEA 导出后替换 `data/fea_cases/*.csv`，保持 `manifest.json` 字段不变；
说明见 `docs/FEA案例与灵敏度.md`。

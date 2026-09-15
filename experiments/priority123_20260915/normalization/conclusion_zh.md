# 归一化敏感性分析结论

分析固定原始 normalized A、图像集合与 CCI 区域；未按 raw 结果重新筛选人群。
raw 与 normalized 的失败率、符号翻转方向、hardest-foil 一致率及固定 normalized B 上的 unrestricted oracle pass capacity 均由完整逐图响应直接计算。
没有将 feasible_set_size 解释为逐候选可行性，也没有从平均响应倒推逐模板结果。

- coco_openai_b16: A=17726; raw failure=64.3180%; normalized failure=64.3687%; sign flip=0.1072% (raw pass→norm fail 0.0790%, raw fail→norm pass 0.0282%); hardest-foil agreement=99.5205%; fixed normalized B=11410, raw/norm oracle pass=6.38036809815951/6.2489044697633656%.
- coco_openai_b32: A=17847; raw failure=64.7616%; normalized failure=64.7784%; sign flip=0.0728% (raw pass→norm fail 0.0448%, raw fail→norm pass 0.0280%); hardest-foil agreement=99.4677%; fixed normalized B=11561, raw/norm oracle pass=7.897240723120837/7.836692327653318%.
- voc2007_openai_b16: A=1209; raw failure=42.1836%; normalized failure=42.2663%; sign flip=0.0827% (raw pass→norm fail 0.0827%, raw fail→norm pass 0.0000%); hardest-foil agreement=99.4210%; fixed normalized B=511, raw/norm oracle pass=27.397260273972602/27.397260273972602%.
- voc2007_openai_b32: A=1227; raw failure=41.1573%; normalized failure=41.1573%; sign flip=0.0000% (raw pass→norm fail 0.0000%, raw fail→norm pass 0.0000%); hardest-foil agreement=99.8370%; fixed normalized B=505, raw/norm oracle pass=31.08910891089109/31.287128712871286%.

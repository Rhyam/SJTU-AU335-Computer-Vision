# 问题包-2
问题包-2的题目大部分与**视差**(**disparity**)相关，建议复习课堂内容熟悉定义后再做作业。

## Files
- 问题包-2.pdf
- Solution.pdf
- rectify.py: 问题5实现代码
- reference: 参考文献

## Remark
- **问题2**：注意区分线段长度和坐标的表示区别，坐标可能为负数。
- **问题3**：问题3题目描述很模糊，按照翻译应该是用视差的定义描述立体重建的**精度**，但助教理解的是描述立体重建的**正确性**即需要**证明视差的定义式**，建议两者都写上防止扣分。
- **问题4**：问题4的证明方法很多，网上可以搜到各种证法，此处只给出了一种。
- **问题5**：按照题目要求阅读*reference/SzeliskiBook_20100903_draft.pdf*的11.1.1后，有两篇论文可以参考：
  - *reference/Computing_rectifying_homographies_for_stereo_vision.pdf*：按照这篇做出来效果更好，但需要自己标定相机参数，不具备条件的可参考另一篇；
  - *reference/Fusiello2000_Article_ACompactAlgorithmForRectificat.pdf*：这篇是根据INRIA-Syntim数据集提供的成对照片和相机参数进行实验，Matlab代码和相机参数均在论文中给出，可轻松复现，Python代码见*rectify.py*。

建议在报告中具体说明算法流程，防止不必要的扣分。

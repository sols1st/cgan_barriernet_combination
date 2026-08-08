paper 文件夹有三篇文章，artical-F122 项目对应的是 Provably 那篇文章，现在我要借用其中的 cgan 部分，使用 cgan 生成的原图像，接入到 BarrierNet，参考 Differentiable 那篇文章，模仿它的框架，但是仿真部分换成 artical-F122 里面的 cgan 进行修改和实验

8.7
现在不使用 cgan 主要问题是无法确保图像到状态之间的可靠性
改进路线：
1. 使用 wgan 还是
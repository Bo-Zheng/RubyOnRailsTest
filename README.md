<h1>时间序列预测</h1>
<h6></h6>本教程是使用 TensorFlow 进行时间序列预测的简介。它构建了几种不同样式的模型，包括卷积神经网络 (CNN) 和循环神经网络 (RNN)。</h6>
<br>
本教程包括两个主要部分，每个部分包含若干小节：
<p>本教程是使用 TensorFlow 进行时间序列预测的简介。它构建了几种不同样式的模型，包括卷积神经网络 (CNN) 和循环神经网络 (RNN)。</p>
<p>本教程包括两个主要部分，每个部分包含若干小节：</p>
<ul>
<li>预测单个时间步骤：
<ul>
<li>单个特征。</li>
<li>所有特征。</li>
</ul></li>
<li>预测多个时间步骤：
<ul>
<li>单次：一次做出所有预测。</li>
<li>自回归：一次做出一个预测，并将输出馈送回模型。</li>
</ul></li>
</ul>
<h2>安裝</h2>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202024-12-18%20221002.png">
<h2>天氣數據集</h2>
<p>本教程使用由马克斯·普朗克生物地球化学研究所记录的天气时间序列数据集</p>
<p>此数据集包含了 14 个不同特征，例如气温、气压和湿度。自 2003 年起，这些数据每 10 分钟就会被收集一次。为了提高效率，您将仅使用 2009 至 2016 年之间收集的数据。数据集的这一部分由 François Chollet 为他的Deep Learning with Python</a> 一书所准备。</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%A4%A9%E6%B0%A3%E6%95%B8%E6%93%9A%E9%9B%861.png">



<h1>时间序列预测</h1>
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
<p>本教程仅处理每小时预测，因此先从 10 分钟间隔到 1 小时对数据进行下采样：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%A4%A9%E6%B0%A3%E6%95%B8%E6%93%9A%E9%9B%862.png">
<p>讓我們看一下數據。是前面幾行:</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%A4%A9%E6%B0%A3%E6%95%B8%E6%93%9A%E9%9B%863.png">
<p>下面是一些特徵隨時間的演變</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%A4%A9%E6%B0%A3%E6%95%B8%E6%93%9A%E9%9B%864.jpg">
<h2>檢查和清理</h2>
<p>接下來，看一下數據集的統計數據:</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%AA%A2%E6%9F%A5%E5%92%8C%E6%B8%85%E7%90%86.jpg">
<h4>風速</h4>
<p>值得注意的一件事是风速 (<code>wv (m/s)</code>) 的 <code>min</code> 值和最大值 (<code>max. wv (m/s)</code>) 列。这个 <code>-9999</code> 可能是错误的。</p>
<p>有一個單獨的風向列，因此速度應大於零(>=0)。將其替換為零"</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E9%A2%A8%E9%80%9F1.png">
<h4>特徵工程</h4>
<p>在潛心建構模型之前，務必了解數據並確保傳遞格式正確的數據</p>
<h4>風</h4>
<p>数据的最后一列 <code>wd (deg)</code> 以度为单位给出了风向。角度不是很好的模型输入：360° 和 0° 应该会彼此接近，并平滑换行。如果不吹风，方向则无关紧要。</p>
<p>現在，風數據的分布狀態如下:</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E9%A2%A8.jpg">
<p>但是，如果將風向和風速列轉換成風向量，模組將更容易解釋:</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E9%A2%A8%E5%90%91%E9%87%8F.png">
<p>模型正確解釋風向量的分布要簡單得多:</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E9%A2%A82.jpg">
<h4>時間</h4>
<p>同樣，Date Time列非常有用，但不是以這種字符串形式。首先將其轉換為秒:</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%99%82%E9%96%931.png">
<p>与风向类似，以秒为单位的时间不是有用的模型输入。作为天气数据，它有清晰的每日和每年周期性。可以通过多种方式处理周期性。</p>
<p>您可以通过使用正弦和余弦变换为清晰的“一天中的时间”和“一年中的时间”信号来获得可用的信号：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%99%82%E9%96%932.jpg">
<p>这使模型能够访问最重要的频率特征。在这种情况下，您提前知道了哪些频率很重要。</p>
<p>如果您没有该信息，则可以通过使用快速傅里叶变换提取特征来确定哪些频率重要。要检验假设，下面是温度随时间变化的。请注意 <code>1/year</code> 和 <code>1/day</code> 附近频率的明显峰值：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%99%82%E9%96%933.jpg">
<img src="">

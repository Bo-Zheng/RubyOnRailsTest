<h1>時間序列預測</h1>
<p>本教程是使用 TensorFlow 進行時間序列預測的簡介。它構建了幾種不同樣式的模型，包括卷積神經網絡 (CNN) 和循環神經網絡 (RNN)。</p>
<p>本教程包括兩個主要部分，每個部分包含若干小節：</p>
<ul>
<li>預測單個時間步驟：
<ul>
<li>單個特徵。</li>
<li>所有特徵。</li>
</ul></li>
<li>預測多個時間步驟：
<ul>
<li>單次：一次做出所有預測。</li>
<li>自回歸：一次做出一個預測，並將輸出饋送回模型。</li>
</ul></li>
</ul>
<h2>安裝</h2>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202024-12-18%20221002.png">
<h2>天氣數據集</h2>
<p>本教程使用由馬克斯·普朗克生物地球化學研究所記錄的天氣時間序列數據集</p>
<p>此數據集包含了 14 個不同特徵，例如氣溫、氣壓和濕度。自 2003 年起，這些數據每 10 分鐘就會被收集一次。為了提高效率，您將僅使用 2009 至 2016 年之間收集的數據。數據集的這一部分由 François Chollet 為他的 Deep Learning with Python</a> 一書所準備。</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%A4%A9%E6%B0%A3%E6%95%B8%E6%93%9A%E9%9B%861.png">
<p>本教程僅處理每小時預測，因此先從 10 分鐘間隔到 1 小時對數據進行下採樣：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%A4%A9%E6%B0%A3%E6%95%B8%E6%93%9A%E9%9B%862.png">
<p>讓我們看一下數據。是前面幾行:</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%A4%A9%E6%B0%A3%E6%95%B8%E6%93%9A%E9%9B%863.png">
<p>下面是一些特徵隨時間的演變</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%A4%A9%E6%B0%A3%E6%95%B8%E6%93%9A%E9%9B%864.jpg">
<h3>檢查和清理</h3>
<p>接下來，看一下數據集的統計數據:</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%AA%A2%E6%9F%A5%E5%92%8C%E6%B8%85%E7%90%86.jpg">
<h4>風速</h4>
<p>值得注意的一件事是風速 (<code>wv (m/s)</code>) 的 <code>min</code> 值和最大值 (<code>max. wv (m/s)</code>) 列。這個 <code>-9999</code> 可能是錯誤的。</p>
<p>有一個單獨的風向列，因此速度應大於零(>=0)。將其替換為零</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E9%A2%A8%E9%80%9F1.png">
<h4>特徵工程</h4>
<p>在潛心構建模型之前，務必了解數據並確保傳遞格式正確的數據</p>
<h4>風</h4>
<p>數據的最後一列 <code>wd (deg)</code> 以度為單位給出了風向。角度不是很好的模型輸入：360° 和 0° 應該會彼此接近，並平滑換行。如果不吹風，方向則無關緊要。</p>
<p>現在，風數據的分佈狀態如下:</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E9%A2%A8.jpg">
<p>但是，如果將風向和風速列轉換成風向量，模組將更容易解釋:</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E9%A2%A8%E5%90%91%E9%87%8F.png">
<p>模型正確解釋風向量的分佈要簡單得多:</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E9%A2%A82.jpg">
<h4>時間</h4>
<p>同樣，Date Time 列非常有用，但不是以這種字符串形式。首先將其轉換為秒:</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%99%82%E9%96%931.png">
<p>與風向類似，以秒為單位的時間不是有用的模型輸入。作為天氣數據，它有清晰的每日和每年週期性。可以通過多種方式處理週期性。</p>
<p>您可以通過使用正弦和餘弦變換為清晰的“一天中的時間”和“一年中的時間”信號來獲得可用的信號：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%99%82%E9%96%932.jpg">
<p>這使模型能夠訪問最重要的頻率特徵。在這種情況下，您提前知道了哪些頻率很重要。</p>
<p>如果您沒有該信息，則可以通過使用快速傅里葉變換提取特徵來確定哪些頻率重要。要檢驗假設，下面是溫度隨時間變化的。請注意 <code>1/year</code> 和 <code>1/day</code> 附近頻率的明顯峰值：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%99%82%E9%96%933.jpg">



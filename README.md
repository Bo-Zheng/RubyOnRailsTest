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
<h3>拆分數據</h3>
<p>您将使用 <code>(70%, 20%, 10%)</code> 拆分出训练集、验证集和测试集。请注意，在拆分前数据没有随机打乱顺序。这有两个原因：</p>
<ol>
<li>确保仍然可以将数据切入连续样本的窗口。</li>
<li>确保训练后在收集的数据上对模型进行评估，验证/测试结果更加真实。</li>
</ol>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%8B%86%E5%88%86%E6%95%B8%E6%93%9A.png">
<h3>歸一化數據</h3>
<p>在训练神经网络之前缩放特征很重要。归一化是进行此类缩放的常见方式：减去平均值，然后除以每个特征的标准偏差。</p>
<p>平均值和标准偏差应仅使用训练数据进行计算，从而使模型无法访问验证集和测试集中的值。</p>
<p>有待商榷的是：模型在训练时不应访问训练集中的未来值，以及应该使用移动平均数来进行此类规范化。这不是本教程的重点，验证集和测试集会确保我们获得（某种程度上）可靠的指标。因此，为了简单起见，本教程使用的是简单平均数。</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%AD%B8%E4%B8%80%E5%8C%96%E6%95%B8%E6%93%9A1.png">
<p>现在看一下这些特征的分布。部分特征的尾部确实很长，但没有类似 <code>-9999</code> 风速值的明显错误。</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%AD%B8%E4%B8%80%E5%8C%96%E6%95%B8%E6%93%9A2.jpg">
<h2>數據窗口化</h2>
<p>本教程中的模型将基于来自数据连续样本的窗口进行一组预测。</p>
<p>输入窗口的主要特征包括：</p>
<ul>
<li>输入和标签窗口的宽度（时间步骤数量）。</li>
<li>它们之间的时间偏移量。</li>
<li>用作输入、标签或两者的特征。</li>
</ul>
<p>本教程构建了各种模型（包括线性、DNN、CNN 和 RNN 模型），并将它们用于以下两种情况：</p>
<ul>
<li>单输出和多输出预测。</li>
<li>单时间步骤和多时间步骤预测。</li>
</ul>
<p>本部分重点介绍实现数据窗口化，以便将其重用到上述所有模型。</p>
<p>根据任务和模型类型，您可能需要生成各种数据窗口。下面是一些示例：</p>
<ol>
<li>例如，要在给定 24 小时历史记录的情况下对未来 24 小时作出一次预测，可以定义如下窗口：</li>
</ol>
<p><img src="https://www.tensorflow.org/static/tutorials/structured_data/images/raw_window_24h.png?hl=zh-cn" alt="对未来 24 小时的一次预测。"></p>
<ol>
<li>给定 6 小时的历史记录，对未来 1 小时作出一次预测的模型将需要类似下面的窗口：</li>
</ol>
<p><img src="https://www.tensorflow.org/static/tutorials/structured_data/images/raw_window_1h.png?hl=zh-cn" alt="对未来 1 小时的一次预测。"></p>
<p>本部分的剩余内容会定义 <code>WindowGenerator</code> 类。此类可以：</p>
<ol>
<li>处理如上图所示的索引和偏移量。</li>
<li>将特征窗口拆分为 <code>(features, labels)</code> 对。</li>
<li>绘制结果窗口的内容。</li>
<li>使用 <a href="https://tensorflow.google.cn/api_docs/python/tf/data/Dataset?hl=zh-cn"><code>tf.data.Dataset</code></a> 从训练、评估和测试数据高效生成这些窗口的批次。</li>
</ol>
<h3>1.索引和偏移量</h3>
<p>首先创建 <code>WindowGenerator</code> 类。<code>__init__</code> 方法包含输入和标签索引的所有必要逻辑。</p>
<p>它还将训练、评估和测试 DataFrame 作为输出。这些稍后将被转换为窗口的 <a href="https://tensorflow.google.cn/api_docs/python/tf/data/Dataset?hl=zh-cn"><code>tf.data.Dataset</code></a>。</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E7%B4%A2%E5%BC%95%E5%92%8C%E5%81%8F%E7%A7%BB%E9%87%8F.jpg">
<P>下面是創建本部分開頭圖表中所示的兩個窗口的代碼:</P>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E7%B4%A2%E5%BC%95%E5%92%8C%E5%81%8F%E7%A7%BB%E9%87%8F2.png">
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E7%B4%A2%E5%BC%95%E5%92%8C%E5%81%8F%E7%A7%BB%E9%87%8F3.png">
<h3>2.拆分</h3>
<p><img src="https://tensorflow.google.cn/static/tutorials/structured_data/images/split_window.png?hl=zh-cn" alt="初始窗口都是连续的样本，这会将其拆分成一个（输入，标签）对"></p>
<p>此图不显示数据的 <code>features</code> 轴，但此 <code>split_window</code> 函数还会处理 <code>label_columns</code>，因此可以将其用于单输出和多输出样本。</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%8B%86%E5%88%861.png">
<P>試試以下代碼:</P>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%8B%86%E5%88%862.png">
<p>通常，TensorFlow 中的数据会被打包到数组中，其中最外层索引是交叉样本（“批次”维度）。中间索引是“时间”和“空间”（宽度、高度）维度。最内层索引是特征。</p>
<p>上面的代码使用了三个 7 时间步骤窗口的批次，每个时间步骤有 19 个特征。它将其拆分成一个 6 时间步骤的批次、19 个特征输入和一个 1 时间步骤 1 特征的标签。该标签仅有一个特征，因为 <code>WindowGenerator</code> 已使用 <code>label_columns=['T (degC)']</code> 进行了初始化。最初，本教程将构建预测单个输出标签的模型。</p>
<h3>3.繪圖</h3>
<p>下面是一個繪圖方法，可已對拆分窗口進行簡單可視化:</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E7%B9%AA%E5%9C%961.jpg">
<p>此繪圖根據項目引用的時間來對齊輸入、標籤和(稍後的)預測:</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E7%B9%AA%E5%9C%962.jpg">
<p>你可以繪製其他列，但是樣本窗口w2配置僅包含T(degC)列的標籤。</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E7%B9%AA%E5%9C%963.jpg">
<h3>創建tf.data.Dataset</h3>
<p>最后，此 <code>make_dataset</code> 方法将获取时间序列 DataFrame 并使用 <a href="https://tensorflow.google.cn/api_docs/python/tf/keras/utils/timeseries_dataset_from_array?hl=zh-cn"><code>tf.keras.utils.timeseries_dataset_from_array</code></a> 函数将其转换为 <code>(input_window, label_window)</code> 对的 <a href="https://tensorflow.google.cn/api_docs/python/tf/data/Dataset?hl=zh-cn"><code>tf.data.Dataset</code></a>。</p>




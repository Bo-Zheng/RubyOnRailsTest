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
<h4>4.創建tf.data.Dataset</h4>
<p>最后，此 <code>make_dataset</code> 方法将获取时间序列 DataFrame 并使用 <a href="https://tensorflow.google.cn/api_docs/python/tf/keras/utils/timeseries_dataset_from_array?hl=zh-cn"><code>tf.keras.utils.timeseries_dataset_from_array</code></a> 函数将其转换为 <code>(input_window, label_window)</code> 对的 <a href="https://tensorflow.google.cn/api_docs/python/tf/data/Dataset?hl=zh-cn"><code>tf.data.Dataset</code></a>。</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%89%B5%E5%BB%BA1.png">
<p><code>WindowGenerator</code> 对象包含训练、验证和测试数据。</p>
<p>使用您之前定义的 <code>make_dataset</code> 方法添加属性以作为 <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset?hl=zh-cn"><code>tf.data.Dataset</code></a> 访问它们。此外，添加一个标准样本批次以便于访问和绘图：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%89%B5%E5%BB%BA2.png">
<p>现在，<code>WindowGenerator</code> 对象允许您访问 <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset?hl=zh-cn"><code>tf.data.Dataset</code></a> 对象，因此您可以轻松迭代数据。</p>
<p><a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset?hl=zh-cn#element_spec"><code>Dataset.element_spec</code></a> 属性会告诉您数据集元素的结构、数据类型和形状。</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%89%B5%E5%BB%BA3.png">
<p>在<code>Dataset</code>上進行迭代會產生具體批次:</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%89%B5%E5%BB%BA4.png">
<h2>單步模型</h2>
<p>基于此类数据能够构建的最简单模型，能够仅根据当前条件预测单个特征的值，即未来的一个时间步骤（1 小时）。</p>
<p>因此，从构建模型开始，预测未来 1 小时的 <code>T (degC)</code> 值。</p>
<p><img src="https://www.tensorflow.org/static/tutorials/structured_data/images/narrow_window.png?hl=zh-cn" alt="预测下一个时间步骤"></p>
<p>配置 <code>WindowGenerator</code> 对象以生成下列单步 <code>(input, label)</code> 对：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%96%AE%E6%AD%A5%E6%A8%A1%E5%9E%8B1.png">
<h3>基線</h3>
<p>在构建可训练模型之前，最好将性能基线作为与以后更复杂的模型进行比较的点。</p>
<p>第一个任务是在给定所有特征的当前值的情况下，预测未来 1 小时的温度。当前值包括当前温度。</p>
<p>因此，从仅返回当前温度作为预测值的模型开始，预测“无变化”。这是一个合理的基线，因为温度变化缓慢。当然，如果您对更远的未来进行预测，此基线的效果就不那么好了。</p>
<p>將輸入發送到輸出</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%9F%BA%E7%B7%9A1.png">
<p>實例畫並評估此模型:</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%9F%BA%E7%B7%9A2.png">
<p>上面的代码打印了一些性能指标，但这些指标并没有使您对模型的运行情况有所了解。</p>
<p><code>WindowGenerator</code> 有一种绘制方法，但只有一个样本，绘图不是很有趣。</p>
<p>因此，创建一个更宽的 <code>WindowGenerator</code> 来一次生成包含 24 小时连续输入和标签的窗口。新的 <code>wide_window</code> 变量不会更改模型的运算方式。模型仍会根据单个输入时间步骤对未来 1 小时进行预测。这里 <code>time</code> 轴的作用类似于 <code>batch</code> 轴：每个预测都是独立进行的，时间步骤之间没有交互：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%9F%BA%E7%B7%9A3.png">
<p>此扩展窗口可以直接传递到相同的 <code>baseline</code> 模型，而无需修改任何代码。能做到这一点是因为输入和标签具有相同数量的时间步骤，并且基线只是将输入转发至输出：</p>
<p><img src="https://www.tensorflow.org/static/tutorials/structured_data/images/last_window.png?hl=zh-cn" alt="对未来 1 小时进行一次预测，每小时一次。"></p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%9F%BA%E7%B7%9A4.png">
<p>通過繪製基線模型的預測值，可以注意到只是標籤向右移動了一小時:</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%9F%BA%E7%B7%9A5.png">
<p>在上面三个样本的绘图中，单步模型运行了 24 个小时。这需要一些解释：</p>
<ul>
<li>蓝色的 <code>Inputs</code> 行显示每个时间步骤的输入温度。模型会接收所有特征，而该绘图仅显示温度。</li>
<li>绿色的 <code>Labels</code> 点显示目标预测值。这些点在预测时间，而不是输入时间显示。这就是为什么标签范围相对于输入移动了 1 步。</li>
<li>橙色的 <code>Predictions</code> 叉是模型针对每个输出时间步骤的预测。如果模型能够进行完美预测，则预测值将直接落在 <code>Labels</code> 上。</li>
</ul>
<h3>線性模型</h3>
<p>可以应用于此任务的最简单的<strong>可训练</strong>模型是在输入和输出之间插入线性转换。在这种情况下，时间步骤的输出仅取决于该步骤：</p>
<p><img src="https://www.tensorflow.org/static/tutorials/structured_data/images/narrow_window.png?hl=zh-cn" alt="单步预测"></p>
<p>没有设置 <code>activation</code> 的 <a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense?hl=zh-cn"><code>tf.keras.layers.Dense</code></a> 层是线性模型。层仅会将数据的最后一个轴从 <code>(batch, time, inputs)</code> 转换为 <code>(batch, time, units)</code>；它会单独应用于 <code>batch</code> 和 <code>time</code> 轴的每个条目。</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E7%B7%9A%E6%80%A7%E6%A8%A1%E5%9E%8B1.png">
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E7%B7%9A%E6%80%A7%E6%A8%A1%E5%9E%8B2.png">
<p>本教成訓練許多模型，因此將訓練過程打包到一個函數中:</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E7%B7%9A%E6%80%A7%E6%A8%A1%E5%9E%8B3.png">
<p>訓練模型並評估其性能:</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E7%B7%9A%E6%80%A7%E6%A8%A1%E5%9E%8B4.png">
<p>与 <code>baseline</code> 模型类似，可以在宽度窗口的批次上调用线性模型。使用这种方式，模型会在连续的时间步骤上进行一系列独立预测。<code>time</code> 轴的作用类似于另一个 <code>batch</code> 轴。在每个时间步骤上，预测之间没有交互。</p>
<p><img src="https://www.tensorflow.org/static/tutorials/structured_data/images/wide_window.png?hl=zh-cn" alt="单步预测"></p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E7%B7%9A%E6%80%A7%E6%A8%A1%E5%9E%8B5.png">
<p>下面是 <code>wide_widow</code> 上它的样本预测绘图。请注意，在许多情况下，预测值显然比仅返回输入温度更好，但在某些情况下则会更差：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E7%B7%9A%E6%80%A7%E6%A8%A1%E5%9E%8B6.png">
<p>線性模型的優點之一是他們相對易於解釋。您可以拉取層的權重，並呈現分配給每個輸入的權重:</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E7%B7%9A%E6%80%A7%E6%A8%A1%E5%9E%8B7.png">
<p>有的模型甚至不會將大多數權重放在輸入<code>T(degC)</code>上。這是隨機初始化的風險之一。</p>
<h3>密集</h3>
<p>在应用实际运算多个时间步骤的模型之前，值得研究一下更深、更强大的单输入步骤模型的性能。</p>
<p>下面是一个与 <code>linear</code> 模型类似的模型，只不过它在输入和输出之间堆叠了几个 <code>Dense</code> 层： </p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%AF%86%E9%9B%861.png">
<h3>多步密集</h3>
<p>单时间步骤模型没有其输入的当前值的上下文。它看不到输入特征随时间变化的情况。要解决此问题，模型在进行预测时需要访问多个时间步骤：</p>
<p><img src="https://www.tensorflow.org/static/tutorials/structured_data/images/conv_window.png?hl=zh-cn" alt="每次预测都使用三个时间步骤。"></p>
<p><code>baseline</code>、<code>linear</code> 和 <code>dense</code> 模型会单独处理每个时间步骤。在这里，模型将接受多个时间步骤作为输入，以生成单个输出。</p>
<p>创建一个 <code>WindowGenerator</code>，它将生成 3 小时输入和 1 小时标签的批次：</p>
<p>请注意，<code>Window</code> 的 <code>shift</code> 参数与两个窗口的末尾相关。</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%AF%86%E9%9B%862.png">
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%AF%86%E9%9B%863.png">
<p>您可以通過添加<code>tf.keras.layers.Flatten作為模型的第一層，在多輸入步驟窗口上訓練<code>dense</code>模型:</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%AF%86%E9%9B%864.png">
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%AF%86%E9%9B%865.png">
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%AF%86%E9%9B%866.png">
<p>此方法的主要缺點是，生成的模型只能在具有此形狀的輸入窗口上執行。</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%AF%86%E9%9B%867.png">
<p>下一部分中的捲積模型將解決這個問題。</p>
<h3>捲積神經網路</h3>
<p>卷积层 (<a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D?hl=zh-cn"><code>tf.keras.layers.Conv1D</code></a>) 也需要多个时间步骤作为每个预测的输入。</p>
<p>下面的模型与 <code>multi_step_dense</code> <strong>相同</strong>，使用卷积进行了重写。</p>
<p>请注意以下变化：</p>
<ul>
<li><a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten?hl=zh-cn"><code>tf.keras.layers.Flatten</code></a> 和第一个 <a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense?hl=zh-cn"><code>tf.keras.layers.Dense</code></a> 替换成了 <a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D?hl=zh-cn"><code>tf.keras.layers.Conv1D</code></a>。</li>
<li>由于卷积将时间轴保留在其输出中，不再需要 <a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/Reshape?hl=zh-cn"><code>tf.keras.layers.Reshape</code></a>。</li>
</ul>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%8D%B2%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF1.png">
<p>在一個樣本批次上運行上述模型，以查看模型是否生成了具有預期形狀的輸出:</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%8D%B2%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF2.png">
<p>在 <code>conv_window</code> 上训练和评估上述模型，它应该提供与 <code>multi_step_dense</code> 模型类似的性能。</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%8D%B2%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF3.png">
<p>此 <code>conv_model</code> 和 <code>multi_step_dense</code> 模型的区别在于，<code>conv_model</code> 可以在任意长度的输入上运行。卷积层应用于输入的滑动窗口：</p>
<p><img src="https://www.tensorflow.org/static/tutorials/structured_data/images/wide_conv_window.png?hl=zh-cn" alt="在序列上执行卷积模型"></p>
<p>如果在较宽的输入上运行此模型，它将生成较宽的输出：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%8D%B2%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF4.png">
<p>请注意，输出比输入短。要进行训练或绘图，需要标签和预测具有相同长度。因此，构建 <code>WindowGenerator</code> 以使用一些额外输入时间步骤生成宽窗口，从而使标签和预测长度匹配： </p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%8D%B2%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF5.png">
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%8D%B2%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF6.png">
<p>现在，您可以在更宽的窗口上绘制模型的预测。请注意第一个预测之前的 3 个输入时间步骤。这里的每个预测都基于之前的 3 个时间步骤：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%8D%B2%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF7.png">
<h3>循環神經網路</h3>
<p>循环神经网络 (RNN) 是一种非常适合时间序列数据的神经网络。RNN 分步处理时间序列，从时间步骤到时间步骤地维护内部状态。</p>
<p>您可以在使用 RNN 的文本生成教程和使用 Keras 的递归神经网络 (RNN) 指南中了解详情。</p>
<p>在本教程中，您将使用称为“长短期记忆网络”(<code>tf.keras.layers.LSTM</code>) 的 RNN 层。</p>
<p>对所有 Keras RNN 层（例如<code>tf.keras.layers.LSTM</code>）都很重要的一个构造函数参数是 <code>return_sequences</code>。此设置可以通过以下两种方式配置层：</p>
<ol>
<li>如果为 <code>False</code>（默认值），则层仅返回最终时间步骤的输出，使模型有时间在进行单个预测前对其内部状态进行预热：</li>
</ol>
<p><img src="https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tutorials/structured_data/images/lstm_1_window.png?raw=true" alt="lstm 预热并进行单一预测"></p>
<ol>
<li>如果为 <code>True</code>，层将为每个输入返回一个输出。这对以下情况十分有用：
<ul>
<li>堆叠 RNN 层。</li>
<li>同时在多个时间步骤上训练模型。</li>
</ul></li>
</ol>
<p><img src="https://www.tensorflow.org/static/tutorials/structured_data/images/lstm_many_window.png?hl=zh-cn" alt="lstm在每个时间步后进行预测"></p>

####################
<p><code>return_sequences=True</code> 时，模型一次可以在 24 小时的数据上进行训练。</p>
<p>注：这将对模型的性能给出悲观看法。在第一个时间步骤中，模型无法访问之前的步骤，因此无法比之前展示的简单 <code>linear</code> 和 <code>dense</code> 模型表现得更好。</p>
####################
####################
####################
<h3>性能</h3>
<p>使用此数据集时，通常每个模型的性能都比之前的模型稍好一些：</p>
####################
####################
<h3>多輸出模型</h3>
<p>到目前为止，所有模型都为单个时间步骤预测了单个输出特征，<code>T (degC)</code>。</p>
<p>只需更改输出层中的单元数并调整训练窗口，以将所有特征包括在 <code>labels</code> (<code>example_labels</code>) 中，就可以将所有上述模型转换为预测多个特征：</p>
####################
<p>请注意，上面标签的 <code>features</code> 轴现在具有与输入相同的深度，而不是 1。</p>
<h4>基線</h4>
<p>此处可以使用相同的基线模型 (<code>Baseline</code>)，但这次重复所有特征，而不是选择特定的 <code>label_index</code>：</p>
####################
####################
<h4>密集</h4>
####################
####################
<h4>RNN</h4>
####################
<<h4>高级：残差连接</h4>
<p>先前的 <code>Baseline</code> 模型利用了以下事实：序列在时间步骤之间不会剧烈变化。到目前为止，本教程中训练的每个模型都进行了随机初始化，然后必须学习输出相较上一个时间步骤改变较小这一知识。</p>
<p>尽管您可以通过仔细初始化来解决此问题，但将此问题构建到模型结构中则更加简单。</p>
<p>在时间序列分析中构建的模型，通常会预测下一个时间步骤中的值会如何变化，而非直接预测下一个值。类似地，深度学习中的<a href="https://arxiv.org/abs/1512.03385" class="external">残差网络</a>（或 ResNet）指的是，每一层都会添加到模型的累计结果中的架构。</p>
<p>这就是利用“改变应该较小”这一知识的方式。</p>
<p><img src="https://www.tensorflow.org/static/tutorials/structured_data/images/residual.png?hl=zh-cn" alt="带有残差连接的模型"></p>
<p>本质上，这将初始化模型以匹配 <code>Baseline</code>。对于此任务，它可以帮助模型更快收敛，且性能稍好。</p>
<p>该方法可以与本教程中讨论的任何模型结合使用。</p>
<p>这里将它应用于 LSTM 模型，请注意 <a href="https://www.tensorflow.org/api_docs/python/tf/keras/initializers/Zeros?hl=zh-cn"><code>tf.initializers.zeros</code></a> 的使用，以确保初始的预测改变很小，并且不会压制残差连接。此处的梯度没有破坏对称性的问题，因为 <code>zeros</code> 仅用于最后一层。</p>
####################
####################
<h4>性能</h4>
<p>以下是这些多输出模型的整体性能。</p>
####################
####################
<p>以上性能是所有模型输出的平均值。</p>
<h2>多步模型</h2>
<p>前几个部分中的单输出和多输出模型都对未来 1 小时进行<strong>单个时间步骤预测</strong>。</p>
<p>本部分介绍如何扩展这些模型以进行<strong>多时间步骤预测</strong>。</p>
<p>在多步预测中，模型需要学习预测一系列未来值。因此，与单步模型（仅预测单个未来点）不同，多步模型预测未来值的序列。</p>
<p>大致有两种预测方法：</p>
<ol>
<li>单次预测，一次预测整个时间序列。</li>
<li>自回归预测，模型仅进行单步预测并将输出作为输入进行反馈。</li>
</ol>
<p>在本部分中，所有模型都将预测<strong>所有输出时间步骤中的所有特征</strong>。</p>
<p>对于多步模型而言，训练数据仍由每小时样本组成。但是，在这里，模型将在给定过去 24 小时的情况下学习预测未来 24 小时。</p>
<p>下面是一个 <code>Window</code> 对象，该对象从数据集生成以下切片：</p>
####################
<h3>基線</h3>
<p>此任务的一个简单基线是针对所需数量的输出时间步骤重复上一个输入时间步骤：</p>
<p><img src="https://www.tensorflow.org/static/tutorials/structured_data/images/multistep_last.png?hl=zh-cn" alt="对每个输出步骤重复最后一次输入"></p>
####################
<p>由于此任务是在给定过去 24 小时的情况下预测未来 24 小时，另一种简单的方式是重复前一天，假设明天是类似的：</p>
<p><img src="https://www.tensorflow.org/static/tutorials/structured_data/images/multistep_repeat.png?hl=zh-cn" alt="重复前一天"></p>
####################
<h3>單次模型</h3>
<p>解决此问题的一种高级方法是使用“单次”模型，该模型可以在单个步骤中对整个序列进行预测。</p>
<p>这可以使用 <code>OUT_STEPS*features</code> 输出单元作为 <code>tf.keras.layers.Dense</code> 高效实现。模型只需要将输出调整为所需的 <code>(OUTPUT_STEPS, features)</code>。</p>
<h4>線性</h4>
<p>基于最后输入时间步骤的简单线性模型优于任何基线，但能力不足。该模型需要根据线性投影的单个输入时间步骤来预测 <code>OUTPUT_STEPS</code> 个时间步骤。它只能捕获行为的低维度切片，可能主要基于一天中的时间和一年中的时间。</p>
<p><img src="https://www.tensorflow.org/static/tutorials/structured_data/images/multistep_dense.png?hl=zh-cn" alt="从上一个时间步骤预测所有时间步骤"></p>
####################
<h4>密集</h4>
<p>在输入和输出之间添加 <code>tf.keras.layers.Dense</code> 可为线性模型提供更大能力，但仍仅基于单个输入时间步骤。</p>
####################
<h4>CNN</h4>
<p>卷积模型基于固定宽度的历史记录进行预测，可能比密集模型的性能更好，因为它可以看到随时间变化的情况：</p>
<p><img src="https://www.tensorflow.org/static/tutorials/structured_data/images/multistep_conv.png?hl=zh-cn" alt="卷积模型查看事物如何随时间变化。"></p>
####################
<h4>RNN</h4>
<p>如果循环模型与模型所做的预测相关，则可以学习使用较长的输入历史记录。在这里，模型将积累 24 小时的内部状态，然后对接下来的 24 小时进行单次预测。</p>
<p>在此单次格式中，LSTM 只需要在最后一个时间步骤上生成输出，因此在 <code>tf.keras.layers.LSTM</code> 中设置 <code>return_sequences=False</code>。</p>
<p><img src="https://www.tensorflow.org/static/tutorials/structured_data/images/multistep_lstm.png?hl=zh-cn" alt="lstm 积累输入窗口的状态，并对未来 24 小时进行一次预测。"></p>
####################
<h3>高级：自回归模型</h3>
<p>上述模型均在单个步骤中预测整个输出序列。</p>
<p>在某些情况下，模型将此预测分解为单个时间步骤可能比较有帮助。 然后，模型的每个输出都可以在每个步骤反馈给自己，并可以根据前一个输出进行预测，就像经典的使用循环神经网络生成序列中介绍的一样。</p>
<p>此类模型的一个明显优势是可以将其设置为生成长度不同的输出。</p>
<p>您可以采用本教程前半部分中训练的任意一个单步多输出模型，并在自回归反馈循环中运行，但是在这里，您将重点关注经过显式训练的模型。</p>
<p><img src="https://www.tensorflow.org/static/tutorials/structured_data/images/multistep_autoregressive.png?hl=zh-cn" alt="将模型的输出反馈到其输入"></p>
<h4>RNN</h4>
<p>本教程仅构建自回归 RNN 模型，但是该模式可以应用于设计为输出单个时间步骤的任何模型。</p>
<p>模型将具有与之前的单步 LSTM 模型相同的基本形式：一个<code>tf.keras.layers.LSTM</code> ，后接一个将 <code>LSTM</code> 层输出转换为模型预测的<code>tf.keras.layers.Dense</code> 层。</p>
<p></code>是封装在更高级 <code>tf.keras.layers.RNN</code> 中的 <code>tf.keras.layers.LSTMCell</code>，它为您管理状态和序列结果（有关详细信息，请参阅使用 Keras 的循环神经网络 (RNN)指南）。</p>
<p>在这种情况下，模型必须手动管理每个步骤的输入，因此它直接将 <code>tf.keras.layers.LSTMCell</code> 用于较低级别的单个时间步骤接口。</p>
####################
####################
<p>该模型需要的第一个方法是 <code>warmup</code>，用来根据输入初始化其内部状态。训练后，此状态将捕获输入历史记录的相关部分。这等效于先前的单步 <code>LSTM</code> 模型：</p>
####################
<p>此方法返回单个时间步骤预测以及 <code>LSTM</code> 的内部状态：</p>
####################
<p>有了 <code>RNN</code> 的状态和初始预测，您现在可以继续迭代模型，并在每一步将预测作为输入反馈给模型。</p>
<p>收集输出预测的最简单方式是使用 Python 列表，并在循环后使用 <code>tf.stack</code>。</p>
<p>注：像这样堆叠 Python 列表仅适用于 Eager-Execution，使用 <code>Model.compile(..., run_eagerly=True)</code> 进行训练，或使用固定长度的输出。对于动态输出长度，您需要使用 <code>tf.TensorArray</code> 代替 Python 列表，并用 <code>tf.range</code> 代替 Python <code>range</code>。</p>
####################
<p>在示例输入上运行此模型：</p>
####################
<p>现在，训练模型：</p>
####################
<h3>性能</h3>
<p>在这个问题上，作为模型复杂性的函数，返回值在明显递减。</p>
####################
<p>本教程前半部分的多输出模型的指标显示了所有输出特征的平均性能。这些性能类似，但在输出时间步骤上也进行了平均。 </p>
####################
<p>从密集模型到卷积模型和循环模型，所获得的增益只有百分之几（如果有的话），而自回归模型的表现显然更差。因此，在<strong>这个</strong>问题上使用这些更复杂的方法可能并不值得，但如果不尝试就无从知晓，而且这些模型可能会对<strong>您的</strong>问题有所帮助。</p>
<h2>後續步驟</h2>
<p>本教程是使用 TensorFlow 进行时间序列预测的简单介绍。</p>
<p>要了解更多信息，请参阅：</p>
<ul>
<li><a href="https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/" class="external">Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow</a>（第 2 版）第 15 章。</li>
<li><a href="https://www.manning.com/books/deep-learning-with-python">Python 深度学习</a>第 6 章。</li>
<li><a href="https://www.udacity.com/course/intro-to-tensorflow-for-deep-learning--ud187" class="external">Udacity 的 Intro to TensorFlow for deep learning</a> 第 8 课，包括<a href="https://github.com/tensorflow/examples/tree/master/courses/udacity_intro_to_tensorflow_for_deep_learning" class="external">练习笔记本</a>。</li>
</ul>
<p>还要记住，您可以在 TensorFlow 中实现任何经典时间序列模型，本教程仅重点介绍了 TensorFlow 的内置功能。</p>





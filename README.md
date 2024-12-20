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







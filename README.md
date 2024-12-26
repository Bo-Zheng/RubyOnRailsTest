<h1>時間序列預測</h1>
南華大學 人工智慧期末報告 11225019 黃泊諍 11225027 陳姵均
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
<p>您將使用 <code>(70%, 20%, 10%)</code> 拆分出訓練集、驗證集和測試集。請注意，在拆分前數據沒有隨機打亂順序。這有兩個原因：</p>
<ol>
<li>確保仍然可以將數據切入連續樣本的窗口。</li>
<li>確保訓練後在收集的數據上對模型進行評估，驗證/測試結果更加真實。</li>
</ol>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%8B%86%E5%88%86%E6%95%B8%E6%93%9A.png">
<h3>歸一化數據</h3>
<p>在訓練神經網絡之前縮放特徵很重要。歸一化是進行此類縮放的常見方式：減去平均值，然後除以每個特徵的標準差。</p>
<p>平均值和標準差應僅使用訓練數據進行計算，從而使模型無法訪問驗證集和測試集中的值。</p>
<p>有待商榷的是：模型在訓練時不應訪問訓練集中的未來值，以及應該使用移動平均數來進行此類規範化。這不是本教程的重點，驗證集和測試集會確保我們獲得（某種程度上）可靠的指標。因此，為了簡單起見，本教程使用的是簡單平均數。</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%AD%B8%E4%B8%80%E5%8C%96%E6%95%B8%E6%93%9A1.png">
<p>現在看一下這些特徵的分佈。部分特徵的尾部確實很長，但沒有類似 <code>-9999</code> 風速值的明顯錯誤。</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%AD%B8%E4%B8%80%E5%8C%96%E6%95%B8%E6%93%9A2.jpg">
<h2>數據窗口化</h2>
<p>本教程中的模型將基於來自數據連續樣本的窗口進行一組預測。</p>
<p>輸入窗口的主要特徵包括：</p>
<ul>
<li>輸入和標籤窗口的寬度（時間步驟數量）。</li>
<li>它們之間的時間偏移量。</li>
<li>用作輸入、標籤或兩者的特徵。</li>
</ul>
<p>本教程構建了各種模型（包括線性、DNN、CNN 和 RNN 模型），並將它們用於以下兩種情況：</p>
<ul>
<li>單輸出和多輸出預測。</li>
<li>單時間步驟和多時間步驟預測。</li>
</ul>
<p>本部分重點介紹實現數據窗口化，以便將其重用到上述所有模型。</p>
<p>根據任務和模型類型，您可能需要生成各種數據窗口。下面是一些示例：</p>
<ol>
<li>例如，要在給定 24 小時歷史記錄的情況下對未來 24 小時作出一次預測，可以定義如下窗口：</li>
</ol>
<p><img src="https://www.tensorflow.org/static/tutorials/structured_data/images/raw_window_24h.png?hl=zh-cn" alt="對未來 24 小時的一次預測。"></p>
<ol>
<li>給定 6 小時的歷史記錄，對未來 1 小時作出一次預測的模型將需要類似下面的窗口：</li>
</ol>
<p><img src="https://www.tensorflow.org/static/tutorials/structured_data/images/raw_window_1h.png?hl=zh-cn" alt="對未來 1 小時的一次預測。"></p>
<p>本部分的剩餘內容會定義 <code>WindowGenerator</code> 類。此類可以：</p>
<ol>
<li>處理如上圖所示的索引和偏移量。</li>
<li>將特徵窗口拆分為 <code>(features, labels)</code> 對。</li>
<li>繪製結果窗口的內容。</li>
<li>使用 <a href="https://tensorflow.google.cn/api_docs/python/tf/data/Dataset?hl=zh-cn"><code>tf.data.Dataset</code></a> 從訓練、評估和測試數據高效生成這些窗口的批次。</li>
</ol>
<h3>1.索引和偏移量</h3>
<p>首先創建 <code>WindowGenerator</code> 類。<code>__init__</code> 方法包含輸入和標籤索引的所有必要邏輯。</p>
<p>它還將訓練、評估和測試 DataFrame 作為輸出。這些稍後將被轉換為窗口的 <a href="https://tensorflow.google.cn/api_docs/python/tf/data/Dataset?hl=zh-cn"><code>tf.data.Dataset</code></a>。</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E7%B4%A2%E5%BC%95%E5%92%8C%E5%81%8F%E7%A7%BB%E9%87%8F.jpg">
<p>下面是創建本部分開頭圖表中所示的兩個窗口的代碼：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E7%B4%A2%E5%BC%95%E5%92%8C%E5%81%8F%E7%A7%BB%E9%87%8F2.png">
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E7%B4%A2%E5%BC%95%E5%92%8C%E5%81%8F%E7%A7%BB%E9%87%8F3.png">
<h3>2.拆分</h3>
<p><img src="https://tensorflow.google.cn/static/tutorials/structured_data/images/split_window.png?hl=zh-cn" alt="初始窗口都是連續的樣本，這會將其拆分成一個（輸入，標籤）對"></p>
<p>此圖不顯示數據的 <code>features</code> 軸，但此 <code>split_window</code> 函數還會處理 <code>label_columns</code>，因此可以將其用於單輸出和多輸出樣本。</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%8B%86%E5%88%861.png">
<p>試試以下代碼：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%8B%86%E5%88%862.png">
<p>通常，TensorFlow 中的數據會被打包到數組中，其中最外層索引是交叉樣本（“批次”維度）。中間索引是“時間”和“空間”（寬度、高度）維度。最內層索引是特徵。</p>
<p>上面的代碼使用了三個 7 時間步驟窗口的批次，每個時間步驟有 19 個特徵。它將其拆分成一個 6 時間步驟的批次、19 個特徵輸入和一個 1 時間步驟 1 特徵的標籤。該標籤僅有一個特徵，因為 <code>WindowGenerator</code> 已使用 <code>label_columns=['T (degC)']</code> 進行了初始化。最初，本教程將構建預測單個輸出標籤的模型。</p>
<h3>3.繪圖</h3>
<p>下面是一個繪圖方法，可已對拆分窗口進行簡單可視化：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E7%B9%AA%E5%9C%961.jpg">
<p>此繪圖根據項目引用的時間來對齊輸入、標籤和（稍後的）預測：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E7%B9%AA%E5%9C%962.jpg">
<p>你可以繪製其他列，但是樣本窗口w2配置僅包含T(degC)列的標籤。</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E7%B9%AA%E5%9C%963.jpg">
<h4>4.創建tf.data.Dataset</h4>
<p>最後，此 <code>make_dataset</code> 方法將獲取時間序列 DataFrame 並使用 <a href="https://tensorflow.google.cn/api_docs/python/tf/keras/utils/timeseries_dataset_from_array?hl=zh-cn"><code>tf.keras.utils.timeseries_dataset_from_array</code></a> 函數將其轉換為 <code>(input_window, label_window)</code> 對的 <a href="https://tensorflow.google.cn/api_docs/python/tf/data/Dataset?hl=zh-cn"><code>tf.data.Dataset</code></a>。</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%89%B5%E5%BB%BA1.png">
<p><code>WindowGenerator</code> 對象包含訓練、驗證和測試數
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%89%B5%E5%BB%BA4.png">
<h2>單步模型</h2>
<p>基於此類數據能夠構建的最簡單模型，能夠僅根據當前條件預測單個特徵的值，即未來的一個時間步驟（1 小時）。</p>
<p>因此，從構建模型開始，預測未來 1 小時的 <code>T (degC)</code> 值。</p>
<p><img src="https://www.tensorflow.org/static/tutorials/structured_data/images/narrow_window.png?hl=zh-tw" alt="預測下一個時間步驟"></p>
<p>配置 <code>WindowGenerator</code> 對象以生成下列單步 <code>(input, label)</code> 對：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%96%AE%E6%AD%A5%E6%A8%A1%E5%9E%8B1.png">
<h3>基線</h3>
<p>在構建可訓練模型之前，最好將性能基線作為與以後更複雜的模型進行比較的點。</p>
<p>第一個任務是在給定所有特徵的當前值的情況下，預測未來 1 小時的溫度。當前值包括當前溫度。</p>
<p>因此，從僅返回當前溫度作為預測值的模型開始，預測「無變化」。這是一個合理的基線，因為溫度變化緩慢。當然，如果您對更遠的未來進行預測，此基線的效果就不那麼好了。</p>
<p>將輸入發送到輸出</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%9F%BA%E7%B7%9A1.png">
<p>實例化並評估此模型:</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%9F%BA%E7%B7%9A2.png">
<p>上面的程式碼打印了一些性能指標，但這些指標並沒有使您對模型的運行情況有所了解。</p>
<p><code>WindowGenerator</code> 有一種繪製方法，但只有一個樣本，繪圖不是很有趣。</p>
<p>因此，創建一個更寬的 <code>WindowGenerator</code> 來一次生成包含 24 小時連續輸入和標籤的窗口。新的 <code>wide_window</code> 變量不會更改模型的運算方式。模型仍會根據單個輸入時間步驟對未來 1 小時進行預測。這裡 <code>time</code> 軸的作用類似於 <code>batch</code> 軸：每個預測都是獨立進行的，時間步驟之間沒有交互：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%9F%BA%E7%B7%9A3.png">
<p>此擴展窗口可以直接傳遞到相同的 <code>baseline</code> 模型，而無需修改任何程式碼。能做到這一點是因為輸入和標籤具有相同數量的時間步驟，並且基線只是將輸入轉發至輸出：</p>
<p><img src="https://www.tensorflow.org/static/tutorials/structured_data/images/last_window.png?hl=zh-tw" alt="對未來 1 小時進行一次預測，每小時一次。"></p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%9F%BA%E7%B7%9A4.png">
<p>通過繪製基線模型的預測值，可以注意到只是標籤向右移動了一小時:</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%9F%BA%E7%B7%9A5.png">
<p>在上面三個樣本的繪圖中，單步模型運行了 24 個小時。這需要一些解釋：</p>
<ul>
<li>藍色的 <code>Inputs</code> 行顯示每個時間步驟的輸入溫度。模型會接收所有特徵，而該繪圖僅顯示溫度。</li>
<li>綠色的 <code>Labels</code> 點顯示目標預測值。這些點在預測時間，而不是輸入時間顯示。這就是為什麼標籤範圍相對於輸入移動了 1 步。</li>
<li>橙色的 <code>Predictions</code> 叉是模型針對每個輸出時間步驟的預測。如果模型能夠進行完美預測，則預測值將直接落在 <code>Labels</code> 上。</li>
</ul>
<h3>線性模型</h3>
<p>可以應用於此任務的最簡單的<strong>可訓練</strong>模型是在輸入和輸出之間插入線性轉換。在這種情況下，時間步驟的輸出僅取決於該步驟：</p>
<p><img src="https://www.tensorflow.org/static/tutorials/structured_data/images/narrow_window.png?hl=zh-tw" alt="單步預測"></p>
<p>沒有設置 <code>activation</code> 的 <a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense?hl=zh-tw"><code>tf.keras.layers.Dense</code></a> 層是線性模型。層僅會將數據的最後一個軸從 <code>(batch, time, inputs)</code> 轉換為 <code>(batch, time, units)</code>；它會單獨應用於 <code>batch</code> 和 <code>time</code> 軸的每個條目。</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E7%B7%9A%E6%80%A7%E6%A8%A1%E5%9E%8B1.png">
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E7%B7%9A%E6%80%A7%E6%A8%A1%E5%9E%8B2.png">
<p>本教程訓練許多模型，因此將訓練過程打包到一個函數中:</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E7%B7%9A%E6%80%A7%E6%A8%A1%E5%9E%8B3.png">
<p>訓練模型並評估其性能:</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E7%B7%9A%E6%80%A7%E6%A8%A1%E5%9E%8B4.png">
<p>與 <code>baseline</code> 模型類似，可以在寬度窗口的批次上調用線性模型。使用這種方式，模型會在連續的時間步驟上進行一系列獨立預測。<code>time</code> 軸的作用類似於另一個 <code>batch</code> 軸。在每個時間步驟上，預測之間沒有交互。</p>
<p><img src="https://www.tensorflow.org/static/tutorials/structured_data/images/wide_window.png?hl=zh-tw" alt="單步預測"></p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E7%B7%9A%E6%80%A7%E6%A8%A1%E5%9E%8B5.png">
<p>下面是 <code>wide_window</code> 上它的樣本預測繪圖。請注意，在許多情況下，預測值顯然比僅返回輸入溫度更好，但在某些情況下則會更差：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E7%B7%9A%E6%80%A7%E6%A8%A1%E5%9E%8B6.png">
<p>線性模型的優點之一是它們相對易於解釋。您可以拉取層的權重，並呈現分配給每個輸入的權重:</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E7%B7%9A%E6%80%A7%E6%A8%A1%E5%9E%8B7.png">
<p>有的模型甚至不會將大多數權重放在輸入<code>T(degC)</code>上。這是隨機初始化的風險之一。</p>
<h3>密集</h3>
<p>在應用實際運算多個時間步驟的模型之前，值得研究一下更深、更強大的單輸入步驟模型的性能。</p>
<p>下面是一個與 <code>linear</code> 模型類似的模型，只不過它在輸入和輸出之間堆疊了幾個 <code>Dense</code> 層：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%AF%86%E9%9B%861.png">
<h3>多步密集</h3>
<p>單時間步驟模型沒有其輸入的當前值的上下文。它看不到輸入特徵隨時間變化的情況。要解決此問題，模型在進行預測時需要訪問多個時間步驟：</p>
<p><img src="https://www.tensorflow.org/static/tutorials/structured_data/images/conv_window.png?hl=zh-cn" alt="每次預測都使用三個時間步驟。"></p>
<p><code>baseline</code>、<code>linear</code> 和 <code>dense</code> 模型會單獨處理每個時間步驟。在這裡，模型將接受多個時間步驟作為輸入，以生成單個輸出。</p>
<p>創建一個 <code>WindowGenerator</code>，它將生成 3 小時輸入和 1 小時標籤的批次：</p>
<p>請注意，<code>Window</code> 的 <code>shift</code> 參數與兩個窗口的末尾相關。</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%AF%86%E9%9B%862.png">
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%AF%86%E9%9B%863.png">
<p>您可以通過添加 <code>tf.keras.layers.Flatten</code> 作為模型的第一層，在多輸入步驟窗口上訓練 <code>dense</code> 模型：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%AF%86%E9%9B%864.png">
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%AF%86%E9%9B%865.png">
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%AF%86%E9%9B%866.png">
<p>此方法的主要缺點是，生成的模型只能在具有此形狀的輸入窗口上執行。</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%AF%86%E9%9B%867.png">
<p>下一部分中的卷積模型將解決這個問題。</p>
<h3>卷積神經網路</h3>
<p>卷積層 (<a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D?hl=zh-cn"><code>tf.keras.layers.Conv1D</code></a>) 也需要多個時間步驟作為每個預測的輸入。</p>
<p>下面的模型與 <code>multi_step_dense</code> <strong>相同</strong>，使用卷積進行了重寫。</p>
<p>請注意以下變化：</p>
<ul>
<li><a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten?hl=zh-cn"><code>tf.keras.layers.Flatten</code></a> 和第一個 <a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense?hl=zh-cn"><code>tf.keras.layers.Dense</code></a> 替換成了 <a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D?hl=zh-cn"><code>tf.keras.layers.Conv1D</code></a>。</li>
<li>由於卷積將時間軸保留在其輸出中，不再需要 <a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/Reshape?hl=zh-cn"><code>tf.keras.layers.Reshape</code></a>。</li>
</ul>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%8D%B2%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF1.png">
<p>在一個樣本批次上運行上述模型，以查看模型是否生成了具有預期形狀的輸出：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%8D%B2%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF2.png">
<p>在 <code>conv_window</code> 上訓練和評估上述模型，它應該提供與 <code>multi_step_dense</code> 模型類似的性能。</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%8D%B2%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF3.png">
<p>此 <code>conv_model</code> 和 <code>multi_step_dense</code> 模型的區別在於，<code>conv_model</code> 可以在任意長度的輸入上運行。卷積層應用於輸入的滑動窗口：</p>
<p><img src="https://www.tensorflow.org/static/tutorials/structured_data/images/wide_conv_window.png?hl=zh-cn" alt="在序列上執行卷積模型"></p>
<p>如果在較寬的輸入上運行此模型，它將生成較寬的輸出：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%8D%B2%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF4.png">
<p>請注意，輸出比輸入短。要進行訓練或繪圖，需要標籤和預測具有相同長度。因此，構建 <code>WindowGenerator</code> 以使用一些額外輸入時間步驟生成寬窗口，從而使標籤和預測長度匹配：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%8D%B2%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF5.png">
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%8D%B2%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF6.png">
<p>現在，您可以在更寬的窗口上繪製模型的預測。請注意第一個預測之前的 3 個輸入時間步驟。這裡的每個預測都基於之前的 3 個時間步驟：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%8D%B2%E7%A9%8D%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF7.png">
<h3>循環神經網路</h3>
<p>循環神經網路 (RNN) 是一種非常適合時間序列資料的神經網路。RNN 分步處理時間序列，從時間步驟到時間步驟地維持內部狀態。</p>
<p>您可以在使用 RNN 的文本生成教程和使用 Keras 的遞歸神經網路 (RNN) 指南中了解詳情。</p>
<p>在本教程中，您將使用稱為“長短期記憶網路”(<code>tf.keras.layers.LSTM</code>) 的 RNN 層。</p>
<p>對所有 Keras RNN 層（例如<code>tf.keras.layers.LSTM</code>）都很重要的一個構造函數參數是 <code>return_sequences</code>。此設置可以通過以下兩種方式配置層：</p>
<ol>
<li>如果為 <code>False</code>（預設值），則層僅返回最終時間步驟的輸出，使模型有時間在進行單個預測前對其內部狀態進行預營：</li>
</ol>
<p><img src="https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tutorials/structured_data/images/lstm_1_window.png?raw=true" alt="lstm 預營並進行單一預測"></p>
<ol>
<li>如果為 <code>True</code>，層將為每個輸入返回一個輸出。這對以下情況十分有用：
<ul>
<li>堆疊 RNN 層。</li>
<li>同時在多個時間步驟上訓練模型。</li>
</ul></li>
</ol>
<p><img src="https://www.tensorflow.org/static/tutorials/structured_data/images/lstm_many_window.png?hl=zh-cn" alt="lstm在每個時間步後進行預測"></p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%BE%AA%E7%92%B0%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF1.png">
<p><code>return_sequences=True</code> 時，模型一次可以在 24 小時的資料上進行訓練。</p>
<p>注：這將對模型的性能給出悲觀看法。在第一個時間步驟中，模型無法訪問之前的步驟，因此無法比之前展示的簡單 <code>linear</code> 和 <code>dense</code> 模型表現得更好。</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%BE%AA%E7%92%B0%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF2.png">
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%BE%AA%E7%92%B0%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF3.png">
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%BE%AA%E7%92%B0%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF4.png">
<h3>性能</h3>
<p>使用此資料集時，通常每個模型的性能都比之前的模型稍好一些：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%80%A7%E8%83%BD.png">
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%80%A7%E8%83%BD2.png">
<h3>多輸出模型</h3>
<p>到目前為止，所有模型都為單個時間步驟預測了單個輸出特徵，<code>T (degC)</code>。</p>
<p>只需更改輸出層中的單元數并調整訓練窗口，以將所有特徵包括在 <code>labels</code> (<code>example_labels</code>) 中，就可以將所有上述模型轉換為預測多個特徵：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%A4%9A%E8%BC%B8%E5%87%BA%E6%A8%A1%E5%9E%8B1.png">
<p>請注意，上面標籤的 <code>features</code> 軸現在具有與輸入相同的深度，而不是 1。</p>
<h4>基線</h4>
<p>此處可以使用相同的基線模型 (<code>Baseline</code>)，但這次重複所有特徵，而不是選擇特定的 <code>label_index</code>：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%A4%9A%E8%BC%B8%E5%87%BA%E6%A8%A1%E5%9E%8B2.png">
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%A4%9A%E8%BC%B8%E5%87%BA%E6%A8%A1%E5%9E%8B3.png">
<h4>密集</h4>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%AF%86%E9%9B%862-1.png">
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%AF%86%E9%9B%862-2.png">
<h4>RNN</h4>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/RNN2-1.png">
<h4>高級：殘差連接</h4>
<p>先前的 <code>Baseline</code> 模型利用了以下事實：序列在時間步驟之間不會劇烈變化。到目前為止，本教程中訓練的每個模型都進行了隨機初始化，然後必須學習輸出相較上一個時間步驟改變較小這一知識。</p>
<p>儘管您可以通過仔細初始化來解決此問題，但將此問題構建到模型結構中則更加簡單。</p>
<p>在時間序列分析中構建的模型，通常會預測下一個時間步驟中的值會如何變化，而非直接預測下一個值。類似地，深度學習中的<a href="https://arxiv.org/abs/1512.03385" class="external">殘差網絡</a>（或 ResNet）指的是，每一層都會添加到模型的累計結果中的架構。</p>
<p>這就是利用“改變應該較小”這一知識的方式。</p>
<p><img src="https://www.tensorflow.org/static/tutorials/structured_data/images/residual.png?hl=zh-cn" alt="帶有殘差連接的模型"></p>
<p>本質上，這將初始化模型以匹配 <code>Baseline</code>。對於此任務，它可以幫助模型更快收斂，且性能稍好。</p>
<p>該方法可以與本教程中討論的任何模型結合使用。</p>
<p>這裡將它應用於 LSTM 模型，請注意 <a href="https://www.tensorflow.org/api_docs/python/tf/keras/initializers/Zeros?hl=zh-cn"><code>tf.initializers.zeros</code></a> 的使用，以確保初始的預測改變很小，並且不會壓制殘差連接。此處的梯度沒有破壞對稱性的問題，因為 <code>zeros</code> 僅用於最後一層。</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/RNN2-2.png">
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/RNN2-3.png">
<h4>性能</h4>
<p>以下是這些多輸出模型的整體性能。</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%80%A7%E8%83%BD2-1.png">
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E6%80%A7%E8%83%BD2-2.png">
<p>以上性能是所有模型輸出的平均值。</p>
<h2>多步模型</h2>
<p>前幾個部分中的單輸出和多輸出模型都對未來 1 小時進行<strong>單個時間步驟預測</strong>。</p>
<p>本部分介紹如何擴展這些模型以進行<strong>多時間步驟預測</strong>。</p>
<p>在多步預測中，模型需要學習預測一系列未來值。因此，與單步模型（僅預測單個未來點）不同，多步模型預測未來值的序列。</p>
<p>大致有兩種預測方法：</p>
<ol>
<li>單次預測，一次預測整個時間序列。</li>
<li>自回歸預測，模型僅進行單步預測並將輸出作為輸入進行反饋。</li>
</ol>
<p>在本部分中，所有模型都將預測<strong>所有輸出時間步驟中的所有特徵</strong>。</p>
<p>對於多步模型而言，訓練數據仍由每小時樣本組成。但是，在這裡，模型將在給定過去 24 小時的情況下學習預測未來 24 小時。</p>
<p>下面是一個 <code>Window</code> 對象，該對象從數據集生成以下切片：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%A4%9A%E6%AD%A5%E6%A8%A1%E5%9E%8B1.png">
<h3>基線</h3>
<p>此任務的一個簡單基線是針對所需數量的輸出時間步驟重複上一個輸入時間步驟：</p>
<p><img src="https://www.tensorflow.org/static/tutorials/structured_data/images/multistep_last.png?hl=zh-cn" alt="對每個輸出步驟重複最後一次輸入"></p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%9F%BA%E7%B7%9A2-1.png">
<p>由於此任務是在給定過去 24 小時的情況下預測未來 24 小時，另一種簡單的方式是重複前一天，假設明天是類似的：</p>
<p><img src="https://www.tensorflow.org/static/tutorials/structured_data/images/multistep_repeat.png?hl=zh-cn" alt="重複前一天"></p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%9F%BA%E7%B7%9A2-2.png">
<h3>單次模型</h3>
<p>解決此問題的一種高級方法是使用“單次”模型，該模型可以在單個步驟中對整個序列進行預測。</p>
<p>這可以使用 <code>OUT_STEPS*features</code> 輸出單元作為 <code>tf.keras.layers.Dense</code> 高效實現。模型只需要將輸出調整為所需的 <code>(OUTPUT_STEPS, features)</code>。</p>
<h4>線性</h4>
<p>基於最後輸入時間步驟的簡單線性模型優於任何基線，但能力不足。該模型需要根據線性投影的單個輸入時間步驟來預測 <code>OUTPUT_STEPS</code> 個時間步驟。它只能捕獲行為的低維度切片，可能主要基於一天中的時間和一年中的時間。</p>
<p><img src="https://www.tensorflow.org/static/tutorials/structured_data/images/multistep_dense.png?hl=zh-cn" alt="從上個時間步驟預測所有時間步驟"></p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%96%AE%E6%AC%A1%E6%A8%A1%E5%9E%8B1.png">
<h4>密集</h4>
<p>在輸入和輸出之間添加 <code>tf.keras.layers.Dense</code> 可為線性模型提供更大能力，但仍僅基於單個輸入時間步驟。</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%96%AE%E6%AC%A1%E6%A8%A1%E5%9E%8B%E5%AF%86%E9%9B%861.png">
<h4>CNN</h4>
<p>卷積模型基於固定寬度的歷史記錄進行預測，可能比密集模型的性能更好，因為它可以看到隨時間變化的情況：</p>
<p><img src="https://www.tensorflow.org/static/tutorials/structured_data/images/multistep_conv.png?hl=zh-cn" alt="卷積模型查看事物如何隨時間變化。"></p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%96%AE%E6%AC%A1%E6%A8%A1%E5%9E%8BCNN1.png">
<h4>RNN</h4>
<p>如果循環模型與模型所做的預測相關，則可以學習使用較長的輸入歷史記錄。在這裡，模型將積累 24 小時的內部狀態，然後對接下來的 24 小時進行單次預測。</p>
<p>在此單次格式中，LSTM 只需要在最後一個時間步驟上生成輸出，因此在 <code>tf.keras.layers.LSTM</code> 中設置 <code>return_sequences=False</code>。</p>
<p><img src="https://www.tensorflow.org/static/tutorials/structured_data/images/multistep_lstm.png?hl=zh-cn" alt="lstm 積累輸入窗口的狀態，並對未來 24 小時進行一次預測。"></p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E5%96%AE%E6%AC%A1%E6%A8%A1%E5%9E%8BRNN.png">
<h3>高級：自回歸模型</h3>
<p>上述模型均在單個步驟中預測整個輸出序列。</p>
<p>在某些情況下，模型將此預測分解為單個時間步驟可能比較有幫助。 然後，模型的每個輸出都可以在每個步驟反饋給自己，並可以根據前一個輸出進行預測，就像經典的使用循環神經網絡生成序列中介紹的一樣。</p>
<p>此類模型的一個明顯優勢是可以將其設置為生成長度不同的輸出。</p>
<p>您可以採用本教程前半部分中訓練的任意一個單步多輸出模型，並在自回歸反饋循環中運行，但是在這裡，您將重點關注經過顯式訓練的模型。</p>
<p><img src="https://www.tensorflow.org/static/tutorial
<p>該模型需要的第一個方法是 <code>warmup</code>，用來根據輸入初始化其內部狀態。訓練後，此狀態將捕獲輸入歷史記錄的相關部分。這等效於先前的單步 <code>LSTM</code> 模型：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E8%87%AA%E5%9B%9E%E6%AD%B8%E6%A8%A1%E5%9E%8BRNN2.png">
<p>此方法返回單個時間步驟預測以及 <code>LSTM</code> 的內部狀態：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E8%87%AA%E5%9B%9E%E6%AD%B8%E6%A8%A1%E5%9E%8BRNN3.png">
<p>有了 <code>RNN</code> 的狀態和初始預測，您現在可以繼續迭代模型，並在每一步將預測作為輸入反饋給模型。</p>
<p>收集輸出預測的最簡單方式是使用 Python 列表，並在循環後使用 <code>tf.stack</code>。</p>
<p>注：像這樣堆疊 Python 列表僅適用於 Eager-Execution，使用 <code>Model.compile(..., run_eagerly=True)</code> 進行訓練，或使用固定長度的輸出。對於動態輸出長度，您需要使用 <code>tf.TensorArray</code> 代替 Python 列表，並用 <code>tf.range</code> 代替 Python <code>range</code>。</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E8%87%AA%E5%9B%9E%E6%AD%B8%E6%A8%A1%E5%9E%8BRNN4.png">
<p>在示例輸入上運行此模型：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E8%87%AA%E5%9B%9E%E6%AD%B8%E6%A8%A1%E5%9E%8BRNN5.png">
<p>現在，訓練模型：</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E8%87%AA%E5%9B%9E%E6%AD%B8%E6%A8%A1%E5%9E%8BRNN6.png">
<h3>性能</h3>
<p>在這個問題上，作為模型複雜性的函數，返回值在明顯遞減。</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E8%87%AA%E5%9B%9E%E6%AD%B8%E6%A8%A1%E5%9E%8B%E6%80%A7%E8%83%BD1.png">
<p>本教程前半部分的多輸出模型的指標顯示了所有輸出特徵的平均性能。這些性能類似，但在輸出時間步驟上也進行了平均。</p>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E8%87%AA%E5%9B%9E%E6%AD%B8%E6%A8%A1%E5%9E%8B%E6%80%A7%E8%83%BD2.png">
<p>從密集模型到卷積模型和循環模型，所獲得的增益只有百分之幾（如果有的話），而自回歸模型的表現顯然更差。因此，在<strong>這個</strong>問題上使用這些更複雜的方法可能並不值得，但如果不嘗試就無從知曉，而且這些模型可能會對<strong>您的</strong>問題有所幫助。</p>
<h2>可能遇到的問題</h2>
<h4>在性能的部份程式碼可能因新舊版本而發生錯誤，如下圖所示:</h4>
<img src="https://github.com/Bo-Zheng/RubyOnRailsTest/blob/main/img/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202024-12-23%20202502.png">
<h4>解決方法</h4>
<h4>使用 <code>metric_name = 'compile_metrics'</code> 來設定指標名稱，並根據此名稱取得 <code>metric_index</code>。</h4>
<h2>後續步驟</h2>
<p>本教程是使用 TensorFlow 進行時間序列預測的簡單介紹。</p>
<p>要了解更多信息，請參閱：</p>
<ul>
<li><a href="https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/" class="external">Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow</a>（第 2 版）第 15 章。</li>
<li><a href="https://www.manning.com/books/deep-learning-with-python">Python 深度學習</a>第 6 章。</li>
<li><a href="https://www.udacity.com/course/intro-to-tensorflow-for-deep-learning--ud187" class="external">Udacity 的 Intro to TensorFlow for deep learning</a> 第 8 課，包括<a href="https://github.com/tensorflow/examples/tree/master/courses/udacity_intro_to_tensorflow_for_deep_learning" class="external">練習筆記本</a>。</li>
</ul>
<p>還要記住，您可以在 TensorFlow 中實現任何經典時間序列模型，本教程僅重點介紹了 TensorFlow 的內建功能。</p>





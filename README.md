# Paper QA - Chemistry

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/77eb0ea1-3e92-4f22-a37a-247ebbc5c85b", alt="paper-qa-gui"/>
    <p>
      <b>Fig 1</b> Paper QA graphical user interface (GUI).
    </p>
  </div>
<br>

## 1.1 What is Paper QA?

**Paper QA** **<sup>1</sup>** is a package designed to help users **query and extract information** from a collection of academic papers by **leveraging large language models (LLMs)** such as those provided by **OpenAI**. 

It utilises a combination of **text embeddings**, **vectorization**, and **LLM-based processing** to deliver accurate and contextually relevant answers to queries, with no hallucinations. The answers to the queries are based on the content of the papers that have been embedded.

## 1.2 How Does Paper QA Work?

### 1.2.1 What are Vectors?

**Vectors** belong to the larger category of **tensors**. In machine learning (ML), "tensor" is used as a generic term for a **single or multi-dimensional array of numbers** in *n*-dimensional space. 

When describing a tensor, the word **dimension** refers to **how many arrays that tensor contains**. When describing a vector, dimension refers to **how many individual numbers/components that vector contains**.

1. A **scalar** is a **zero-dimensional tensor**, containing a **single number**.
   * For example, a system modelling weather data might represent a single day's high temperature (in Celsius) in **scalar form** as 33
2. A **vector** is a **one-dimensional tensor**, containing **multiple scalars** of the **same type of data**.
   * For example, the weather model might represent the low, mean and high temperatures of that single day in **vector form** as (25, 30, 33). Each scalar component is a **feature (i.e. dimension)** of the vector, corresponding to a feature of that day's weather.
3. A **tuple** is a **one-dimensional tensor**, containing **multiple scalars** of **more than one type of data**.
   * For example, a person’s name, age and height (in inches) might be represented in tuple form as (Jane, Smith, 31, 65).
4. A **matrix** is a **two-dimensional tensor**, containgin **multiple vectors of the same type of data**. Intuitively, it can be visualised as a **two-dimensional array/grid of scalars**, where **each row or column is a vector**.
   * For example, that weather model might represent the entire month of June as a 3x30 matrix, in which each row is a feature vector describing an individual day’s low, mean and high temperatures.

### 1.2.2 What are Embeddings?

Though the terms are often used interchangably in ML, **vectors and embeddings are not the same thing**.

An **embedding** is **any numerical representations** of data that **captures its relevant qualities** in a way that **ML algorithms can process**. The data is **embedded in n-dimensional space**.

In theory, data doesn't need to be 


### 1.2.3 What are Vector Embeddings?

Vector embeddings are **numerical representations of data points**, and can include **non-mathematical data** such as **words** or **images**. An embedding is a **vector (a multi-dimensional array of floating points)**, and the **distance between two vectors** measures their **relatedness**. **Small distances** suggest **high relatedness**, and **large distances** suggest **low relatedness**.

Any data that an AI model operates on, including **non-mathematical, unstructured data** such as text, audio or images, **must be expressed numerically**. At a high level, **vector embedding** 


## References

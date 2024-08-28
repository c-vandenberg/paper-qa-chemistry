# Paper QA - Chemistry

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/77eb0ea1-3e92-4f22-a37a-247ebbc5c85b", alt="paper-qa-gui"/>
    <p>
      <b>Fig 1</b> Custom Paper QA Chemistry graphical user interface (GUI).
    </p>
  </div>

## 1.1 What is Paper QA?

[**Paper QA**](https://github.com/Future-House/paper-qa) is a package designed to help users **query and extract information** from a collection of academic papers by **leveraging large language models (LLMs)** such as those provided by **OpenAI**. 

It utilises a combination of **text embeddings**, **vectorization**, and **LLM-based processing** to deliver accurate and contextually relevant answers to queries, with no hallucinations. The answers to the queries are based on the content of the papers that have been embedded, along with **prompts** used to contextualise the answer.

## 1.2 Paper QA Chemistry Implementation

### 1.2.1 Features

1. This program builds on the aforementioned Paper QA package to extract papers from a user's **Zotero database**. Specifically, it has been designed using the author's database of academic papers in the areas of:
   * **Organic chemistry**
   * **Drug discovery and development**
   * **Cheminformatics**
   * And the **applications of machine learning to these areas**.
2. It **overrrides some of the logic** in `paperqa.contrib.ZoteroDB`. This was necessary as it was discovered during the development that papers past the first 100 in the Zoetero database were not being processed, even if the starting position was set as >100.
3. It allows the user to **choose the LLM to be used**, how many papers to embed, and where in the database to start the processing batch, and input their query. It additionally outputs embedding information (e.g. embedding progress, number of tokens per paper etc.)
4. It also adds the feature of **pickling the `Docs` object** to a **`.pkl` file**, maintaining the **state** of a the `Docs` object for future runs of the program. One `.pkl` file is generated per LLM used.
5. Finally, it wraps the bespoke Paper QA application inside `PySimpleGUI` (**Fig 1**).

### 1.2.2 Usage

Although Paper QA Chemistry was designed to use the author's Zotero database, it can be **used with any Zotero database**. After cloning this repository, the user will simply need to provide the **following environment variables** (provided they have a **Zotero account** and an **OpenAI API key with credit**):
1. `ZOTERO_USER_ID`
2. `ZOTERO_API_KEY`
3. `OPENAI_API_KEY` 

These can be added either to the project's `.env` file, or directly to the IDE.

Additionally, the dependencies defined in `requirements.txt` require Python 3.10.

**N.B.** Although developed in **Ubuntu 24.04 LTS**, it has been tested and modified for use in Windows 11.

### 1.2.3 Examples

In **Fig 2** you can see the user interface when embedding further papers into the `Docs` object. The user can specify **how many papers to embed** (maximum batch size of 100 due to Zotero API limits), and **where in the database to start the embedding**. Using both of these the user can embed their **entire library** if needed (functionality to streamline this is being developed). 

The user is also given the progress of the embedding, and how many tokens each paper contains, giving a crude estimate of **how much the paper will cost to process**. The **exact cost is not given** as this depends on both the embedding model and LLM used.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/5386a037-6ea1-443e-9252-717ab7d821cb", alt="paper-qa-embedding-example"/>
    <p>
      <b>Fig 2</b> Paper QA Chemistry example embedding.
    </p>
  </div>

In **Fig 3**, you can see the answer to the query: 

> **Regarding machine learning in chemistry, what are some recent advances in the areas of retrosynthesis, reaction prediction, and reaction optimisation?**

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/97f02fc5-1780-417f-bcbb-e706cb9b2e5d", alt="paper-qa-example-query-answer"/>
    <p>
      <b>Fig 3</b> Paper QA Chemistry example query answer.
    </p>
  </div>

## 1.3 Vectors, Embeddings, and Vector Embeddings

Before we discuss how Paper QA works, it is essential to understand the concepts of **vectors**, **embeddings**, and **vector embeddings**. If the reader understands these topics already, feel free to skip to sections **1.4** and **1.4**.

### 1.3.1 What are Vectors?

**Vectors** belong to the larger category of **tensors**. In machine learning (ML), "tensor" is used as a generic term for a **single or multi-dimensional array of numbers** in *n*-dimensional space. **<sup>1</sup>**

When describing a tensor, the word **dimension** refers to **how many arrays that tensor contains**. When describing a vector, dimension refers to **how many individual numbers/features that vector contains**.

1. A **scalar** is a **zero-dimensional tensor**, containing a **single number**.
   * For example, a system modelling weather data might represent a single day's high temperature (in Celsius) in **scalar form** as 33
2. A **vector** is a **one-dimensional tensor**, containing **multiple scalars** of the **same type of data**.
   * For example, the weather model might represent the low, mean and high temperatures of that single day in **vector form** as (25, 30, 33). Each scalar component is a **feature (i.e. dimension)** of the vector, corresponding to a feature of that day's weather.
3. A **tuple** is a **one-dimensional tensor**, containing **multiple scalars** of **more than one type of data**.
   * For example, a person’s name, age and height (in inches) might be represented in tuple form as (Jane, Smith, 31, 65).
4. A **matrix** is a **two-dimensional tensor**, containgin **multiple vectors of the same type of data**. Intuitively, it can be visualised as a **two-dimensional array/grid of scalars**, where **each row or column is a vector**.
   * For example, that weather model might represent the entire month of June as a 3x30 matrix, in which each row is a feature vector describing an individual day’s low, mean and high temperatures. **<sup>1</sup>**

### 1.3.2 What are Embeddings?

Though the terms are often used interchangably in ML, **vectors and embeddings are not the same thing**.

An **embedding** is **any numerical representations** of data that **captures its relevant qualities** in a way that **ML algorithms can process**. The data is **embedded in n-dimensional space**.

In theory, data **doesn't need to be embedded as a vector**. For example, some types of data can be embedded in **tuple form**. However in practice, **embeddings predominantely take the form of vectors in modern ML**. **<sup>1</sup>**

### 1.3.3 What are Vector Embeddings?

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/cd7c2ab4-4237-4808-9445-5be686e8f7e2", alt="vector-embeddings"/>
    <p>
      <b>Fig 4</b> Schematic representation of vector embedding. <b><sup>2</sup></b>
    </p>
  </div>

Vector embeddings are **numerical representations of data points**, and can include **non-mathematical data** such as **words** or **images**. 

Vector embedding **transforms a data point** into an **n-dimensional array of floating point numbers** representing that data point's **characteristics** (i.e. its **features**). Vector embeddings can have **dozens, hundreds or even thousands of dimensions**. 

Vector embedding is achieved by **training an embedding model** on an **data set relevant to the task at hand** or by using a **pretrained embedding model**. is a **vector (a multi-dimensional array of floating points)**, and the **distance between two vectors** measures their **relatedness**: 
* **Small distances** suggest **high relatedness**, and **large distances** suggest **low relatedness**. **<sup>1</sup>**

### 1.3.4 How to Compare Vector Embeddings?

Any data that an AI model operates on, including **non-mathematical, unstructured data** such as text, audio or images, **must be expressed numerically**. At a high level, **vector embedding** is a way to **convert an unstructured data point** into an **array of numbers** that **still expresses that data's original meaning**.

The **core logic of vector embeddings** is that **n-dimensional embeddings of similar data points** should be **grouped closely together in n-dimensional space**. That is, the **distance between two vectors** measures their **relatedness**. **Small distances** suggest **high relatedness**, and **large distances** suggest **low relatedness**. **<sup>1</sup>**

## 1.4 Vector Embedding in Paper QA

In Paper QA, both the **academic papers** and **user queries** are **converted into embeddings**. This allows the system to **compare and analyse the semantic content of the query** against the **vast amount of information contained within the papers**.

As stated previously, the key idea is that **semantically similar pieces of text** will have **embeddings that are close to each other in the vector space**. In Paper QA, these embeddings can be:
1. **Word Embeddings** - This is when **individual words** are represented as vectors, where **words with similar meanings have similar vectors**
2. **Sentence or Document Embeddings** - This **extends the concept to entire sentences or documents**, creating a **single vector** that **captures the overall meaning of the text**.

## 1.5 How Does Paper QA Work?

### 1.5.1 Vector Embedding of Academic Papers

Paper QA uses the following process to embed the academic papers into vectors:

1. **Text Extraction**: The text from the paper to be embedded to extracted, typically **section by section** (e.g. abstract, introduction, methods etc).
2. **Text Tokenization**: **Text generation** and **embedding** models process text in **chunks** called **tokens**. Therefore, the words in the text are **decomposed into these tokens**, with 1 token being approximately **4 characters or 0.75 words** for English text.
3. **Vector Embedding**: Each token is then **passed through a pre-trained embedding model** to convert it into a vector. By default Paper QA uses OpenAI's **`text-embedding-3-small`** embedding model.
4. **Vector Storage**: The resulting vectors are then **stored in a vector database**. This allows for **efficient retrieval and comparison when a query is made**. Paper QA uses a **simple numpy vector database** to embed and search documents.

### 1.5.2 Vector Embedding of Query

When a user submits a query, Paper QA follows a similar process:
1. **Query Preprocessing**: The query is first **processed** to **remove any unnecessary information** and to **prepare it for embedding**.
2. **Query Embedding**: The processed query then follows the **same process as the academic papers**, with the **same embedding model**, generating a vector that represents the **semantic meaning of the query**.

### 1.5.3 Matching the Query with Relevant Papers

Once the query has been embedded into a vector, Paper QA **compares this vector** with the **vectors of the paper tokens stored in the database**. This involves:

1. **Similarity Search**: The system performs a similarity search between the **query vector** and the **paper vectors**. This involves calculating the **cosine similarity between the vectors**. This is a measure of **how close the vectors are to each other in the vector space**. Paper vectors that are **closer to the query vector** represent paper text that is **more semantically similar** to the query.
2. **Relevant Text Retrieval**: The tokens (which may represent single words, sentences or entire sections) that are **most similar to the query** are retrieved, ranked and passed to the LLM for **further processing**.

### 1.5.4 LLM Answer Generation

The final step involves **generating an answer to the user's query**:

1. **Re-scoring**: The **retrieved paper tokens**, which are **most relevant to the query**, are **fed into an LLM** to be re-scored and the text summarised. By default, Paper QA uses `gpt-4o-mini` for this step.
2. **Contextual Answering with Prompt**: The system has a defined prompt that can be used to contextualise the answer to the query. The summarised text is then put into the prompt and **fed into another LLM**. By default, Paper QA uses `gpt-4-turbo` for this step. The LLM uses this context to generate a **coherent and accurate answer** to the query.
3. **Reference and Source Integration**: The LLM can also be configured via the prompt to **provide references to the papers** or the **sections of papers** it used to generate the answer.

## References

**[1]** Bergmann, D. and Stryker, C. (2024) What is vector embedding?, IBM. Available at: https://www.ibm.com/think/topics/vector-embedding (Accessed: 22 August 2024). <br><br>
**[2]** Tripathi, R. (no date) What are vector embeddings, Pinecone. Available at: https://www.pinecone.io/learn/vector-embeddings/ (Accessed: 22 August 2024). <br><br>

I quick project to explore Large Language Models using OpenAI's API: https://platform.openai.com/docs/api-reference.

Topics Explored:
- Chat Completions API
- Embeddings API Endpoint
- Argubots / Agents:
  - LLM Argubot
  - LLM Agents
  - Simulating Human Characters
  - Simulating Conversation
  - Batched Generation
  - Model-Based Evaluation
  - Similarity-Based Retrieval
    - Kialo Corpus
    - BM25
  - Retrieval Argubot
  - Retrieval-Augmented Generation (RAG)
  - Prompt Engineering
  - Chain of Thought / Planning
  - Few-Shot Prompting

## Overview of Retrieval-Augmented Generation (RAG)

1. **Query formation step**: Ask the LLM what claim should be responded to.  For
   example, consider the following dialogue:
    > ...
    > Aragorn: Fortunately, the vaccine was developed in record time.
    > Human: Sounds fishy.

    "Sounds fishy" is exactly the kind of statement that Akiko had trouble using
    as a Kialo query.  But Aragorn shows the *whole dialogue* to the LLM, and
    asks the LLM what the human's *last turn* was really saying or implying, in
    that context. The LLM answers with a much longer statement:

    > Human [paraphrased]: A vaccine that was developed very quickly cannot be trusted.
    > If its developers are claiming that it is safe and effective, I question their motives.

    This paraphrase makes an explicit claim and can be better understood without the context.
    It also contains many more word types, which makes it more likely that BM25 will be able
    to find a Kialo claim with a nontrivial number of those types. 

2. **Retrieval step**: Look up claims in Kialo that are similar to the explicit
   claim.  Create a short "document" that describes some of those claims and
   their neighbors on Kialo.

3. **Retrieval-augmented generation**: Prompt the LLM to generate the response
   (like any `LLMAgent`).  But include the new document somewhere in the LLM
   prompt, in a way that it influences the response. 
   
   Thus, the LLM can respond in a way that is appropriate to the dialogue but
   also draws on the curated information that was retrieved in Kialo.  After
   all, it is a Transformer and can attend to both!

## RAG in Action

The goal is to generate an argument against the last claim in dialogue d.

### Query Formation Step

We utilize the LLM to find the explicit claim from the implicit meaning. We can utilize few-shot prompting as shown to improve the query. A further improvement is to directly take explicit claims from the Kialo Corpus in our few-shot prompting to guide the LLM to generate responses similar to the corpus.

<img src="https://github.com/KiwiThePoodle/Large-Language-Models/blob/main/Query%20Formation%20Step.png">

### Retrieval Step

We utilize the Kialo Corpus and BM25 to find a similar explicit claim. We then create a description of the claim including arguments in favor of and against that claim.

<img src="https://github.com/KiwiThePoodle/Large-Language-Models/blob/main/Retrieval%20Step.png">

### Retrieval-Augmented Generation

We prompt the LLM to generate an appropriate response based on the description. We can utilize prompt engineering to improve the responses for our evaluation.

<img src="https://github.com/KiwiThePoodle/Large-Language-Models/blob/main/Retrieval-Augmented%20Generation.png">

### Result

Aragorn: RAG Agent (More advanced LLM Agent)

Alice: LLM Agent

Dialogue:

(Alice) What are your thoughts on the importance of corporations taking responsibility for their environmental impact?

(Aragorn) Acting proactively to address global challenges such as energy, water, and food security is not only possible, but also morally imperative. Waiting for crises to force change is a dangerous and irresponsible approach. By taking proactive measures, such as investing in renewable energy, implementing sustainable water management practices, and promoting ethical and efficient agricultural methods, we can work towards mitigating the severity of potential crises and creating a more stable and equitable world for future generations. It is our ethical duty to strive for solutions that prevent suffering and promote the well-being of all people, rather than waiting for crises to force change.

(Alice) Absolutely, proactivity is important in addressing global challenges. However, it's also crucial to understand the challenges and complexities involved in implementing large-scale changes, as well as the potential impact on various stakeholders, including economic repercussions for certain industries and regions. Balancing proactive measures with consideration for all those affected can be a intricate challenge.

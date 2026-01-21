<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->


<h1 align="center">Surgery: Mitigating Harmful  Fine-Tuning for Large Language Models via Attention Sink</h1>



Harmful fine-tuning can invalidate safety alignment of large language models, exposing significant safety risks. In this paper, we utilize the attention sink mechanism to mitigate harmful fine-tuning. Specifically, we first measure a statistic named sink divergence for each attention head and observe that different attention heads exhibit two different signs of sink divergence. To understand its safety implications, we conduct experiments and find that the number of attention heads of positive sink divergence increases along with the increase of the model's harmfulness when undergoing harmful fine-tuning. Based on this finding, we propose a separable sink divergence hypothesis -- attention heads associating with learning harmful patterns during fine-tuning are separable by their sign of sink divergence. Based on the hypothesis, we propose a fine-tuning-stage defense, dubbed Surgery. Surgery utilizes a regularizer for sink divergence suppression, which steers attention heads toward the negative sink divergence group, thereby reducing the model’s tendency to learn and amplify harmful patterns. Extensive experiments demonstrate that Surgery improves defense performance by 5.90%, 11.25%, and 9.55% on the BeaverTails, HarmBench, and SorryBench benchmarks, respectively. 

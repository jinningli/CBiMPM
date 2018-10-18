# CBiMPM: Convolutional Bilateral Multi-Perspective Matching for Natural Language Inference
Course Project in CS229 Natural Language Processing.

### Abstract
The bilateral multi-perspective matching (BiMPM) model achieves a high performance on the `SNLI dataset.` Given two sentences P and Q, BiMPM first encodes them with a Bi-LSTM encoder. Next, it match the two encoded sentences in two directions P against H and H against P. In each matching direction, each time step of one sentence is matched against all timesteps of the other sentence from multiple perspectives. Then, another Bi-LSTM layer is utilized to aggregate the matching results into a fixed-length matching vector. Finally, based on the matching vector, a decision is made through a fully connected layer. This work inspires me to use the matching between P and H to better capture the relationshp between them. An accuracy of `86.4%` is achieved by BiMPM.

Based on the idea of BiMPM, I propose a new architecture named `Convolutional BiMPM (CBiMPM)`. The sequences of P and Q produced by Bi-LSTM layer are designed to go through a convolutional layer and a max-pooling layer. Then, a convolutional interaction between them is applied. The output of max-pooling layer and the interaction vecter are combined with the result of BiMPM before the fully connected layer. Experiments prove CBiMPM achieves a best accuracy of `86.7%` under the same hyper-parameters and experi- ment environment.

### Reference
```
Wang, Zhiguo, Wael Hamza, and Radu Florian. "Bilateral multi-perspective matching for natural language sentences." arXiv preprint arXiv:1702.03814 (2017).
```

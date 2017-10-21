#Cracking_Substitution_Ciphers_Using_Hidden_Markov_Models_And_Digraph_Statistics

In my previous project I have used Hidden Markov Models in determining the structure of English language by specifying various "hidden" states.

One good application of HMM's is using it to crack substitution ciphers based on digraph statistics of English text.

We use digraph statistics (i.e. the frequency of one letter following the other) from some plain text in order to create a frequency matrix. We will use this matrix as our A matrix.

 We initialize our B and pi matrices using the same logic as before (random values which are close to uniform and are row stochastic).

Our observation sequence would be the encrypted text from brown.txt file.

Using this information we train our HMM and gain some valuable information from the final B matrix.
 
Note that we do not re-estimate the A matrix in this case since we want to maintain the digraph properties of the original plain text.

Observing the B matrix would tell us the plain - cipher text mappings. These mapings are derived by calculating the highest probability in each row of the B matrix.

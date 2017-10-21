#include <stdlib.h> 
#include <stdio.h> 
#include <math.h> 
#include <float.h> 
#include <math.h> 
#include <unistd.h> 
#define N 26
#define M 26
#define T 1000
#define K 4

int getIndex(char c) {
    int index = (c - 'a');
    return index;
}

int main() {
    printf("\nCracking Substitution Cipher using Digraph based HMM...\n\n");
    sleep(1);
    double A[N][N];
    double rowsum[N];
    double tempB[N][M];
    double B[N][M];
    double tempPi[N];
    double pi[N];
    double sum = 0;
    /* 
       We initialize the A matrix in this program using the diagraph statistics of 
       English text. We read the sme brown.txt file (first 1000000 characters) in order
       to establish the digraph statistics. 
    */
    printf("Creating A matrix based on diagraph statistics of brown.txt...\n\n");
    sleep(1);
    int count['z' - 'a' + 1]['z' - 'a' + 1] = {
        {
            0
        }
    };
    int char0 = EOF, char1, char2;
    int t = 1000000, tcnt = 0;
    printf("Reading plain-text file for digraph matrix...\n\n");
    sleep(1);
    FILE * file;
    file = fopen("brown.txt", "r");
    char1 = getc(file);
    if (file != NULL) {
        printf("File present at specified location...\n\n");
        sleep(1);
        while ((char2 = getc(file)) != EOF && tcnt < t) {
            count[char1 - 'a'][char2 - 'a']++;
            char1 = char2;
            tcnt++;
        }
        fclose(file);
        for (char1 = 'a'; char1 <= 'z'; char1++) {
            for (char2 = 'a'; char2 <= 'z'; char2++) {
                int fcount = count[char1 - 'a'][char2 - 'a'];
                A[getIndex(char1)][getIndex(char2)] = fcount;
            }
        }
    }
    printf("Digraph matrix created...\n\n");
    sleep(1);
    /* 
       Add 5 to each value and then divide by row sum.
       Values turn out to be row stochastic. 
    */
    for (int i = 0; i < 26; i++) {
        for (int j = 0; j < 26; j++) {
            A[i][j] = A[i][j] + 5;
        }
    }
    double sumA = 0;
    for (int i = 0; i < 26; i++) {
        for (int j = 0; j < 26; j++) {
            sumA = sumA + A[i][j];
        }
        rowsum[i] = sumA;
        sumA = 0;
    }
    /* Normalize the matrix */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = A[i][j] / rowsum[i];
        }
    }
    /* 
       Initialize the B matrix with close to normal values, i.e. 1/N.
       Add 5 to each value and then divide by row sum.
       Values turn out to be row stochastic. 
    */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            sum = 0;
            double scale = rand() / (double) RAND_MAX; /* [0, 1.0] */
            tempB[i][j] = 0 + scale * (1 - 0);
            sum += tempB[i][j];
            for (int i = 0; i < N; i++) {

                for (int j = 0; j < M; j++) {

                    B[i][j] = tempB[i][j] / sum;
                }
            }
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            B[i][j] = B[i][j] + 5;
        }
    }
    double sumB = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            sumB = sumB + B[i][j];
        }
        rowsum[i] = sumB;
        sumB = 0;
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            B[i][j] = B[i][j] / rowsum[i];
        }
    }
    /* Initialize the pi matrix with close to normal values, i.e. 1/N.
       Add 5 to each value and then divide by row sum.
       Values turn out to be row stochastic. 
    */
    for (int j = 0; j < N; j++) {
        sum = 0;
        double diff = 0.6 - 0.4;
        tempPi[j] = (((float) rand() / RAND_MAX) * diff) + 0.4;
        sum += tempPi[j];
    }
    for (int j = 0; j < N; j++) {
        pi[j] = tempPi[j] / sum;
    }
    for (int j = 0; j < N; j++) {

        pi[j] = pi[j] + 5;

    }
    double sumPi = 0;
    for (int j = 0; j < N; j++) {
        sumPi = sumPi + pi[j];
    }
    for (int j = 0; j < N; j++) {
        pi[j] = pi[j] / sumPi;
    }
    int O[T];
    int maxIterations = 200;
    int iters = 0;
    float oldLogProb = -DBL_MAX;
    float c[T];
    float alpha[T][N];
    float beta[T][N];
    float gamma[T][N];
    float digamma[T][N][N];
    /* We encrypt the plain text in order to initialize our observation sequence array. */
    int tcount = 0;
    int position = 0;
    char plaintext;
    char ciphertext;
    FILE * fptr;
    fptr = fopen("brown.txt", "r");
    if (fptr) {
        printf("Reading plain-text file for encryption and observation sequence creation...\n\n");
        sleep(1);
        printf("Creating Observation Sequence from File...\n\n");
        sleep(1);
        while ((plaintext = fgetc(fptr)) != EOF && tcount < T) {
            position = plaintext + K;
            if (position > 'z')
                position = position - 26;
            ciphertext = position;
            O[tcount] = getIndex(ciphertext);
            tcount++;
        }
    }
    fclose(fptr);
    /* Initialize alpha, beta, gamma, digamma and the scale array to 0. */
    for (int i = 0; i < T; i++)
        for (int j = 0; j < N; j++)
            alpha[i][j] = 0;
    for (int i = 0; i < T; i++)
        for (int j = 0; j < N; j++)
            beta[i][j] = 0;
    for (int i = 0; i < T; i++)
        for (int j = 0; j < N; j++)
            gamma[i][j] = 0;
    for (int i = 0; i < T; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                digamma[i][j][k] = 0;
    for (int i = 0; i < T; i++)
        c[i] = 0;
    /* 
       This portion of code is where we actually train the HMM model, which is defined as λ = (A, B, pi).
       The model is trained and the B and pi matrices are re-estimated every iteration until certain end conditions are met.
       This algorithm is also called the Baum-Welsch re-estimation algorithm. 
       Notice that we do not re-estimate the A matrix in this problem, since we want the diagraph staistics to remain intact.
       The final B matrix would reveal information regarding the mapping for alphabets based on the encryption key that we used. 
    */
    printf("Training the Hidden Markov Model on given Observation Sequence...\n\n");
    sleep(1);
    alpha:
        /* 
           This is the alpha-pass or also known as the forward algorithm.
           This algorithm is used for scoring sequences.
           It should be noted that final alpha-pass values are scaled using the scale array. 
        */
        for (int i = 0; i < N; i++) {
            alpha[0][i] = pi[i] * B[i][O[0]];
            c[0] = c[0] + alpha[0][i];
        }
    c[0] = 1 / c[0];
    for (int i = 0; i < N; i++) {
        alpha[0][i] = c[0] * alpha[0][i];
    }
    for (int t = 1; t < T; t++) {
        c[t] = 0;
        for (int i = 0; i < N; i++) {
            alpha[t][i] = 0;
            for (int j = 0; j < N; j++) {
                alpha[t][i] = alpha[t][i] + alpha[t - 1][j] * A[j][i];
            }
            alpha[t][i] = alpha[t][i] * B[i][O[t]];
            c[t] = c[t] + alpha[t][i];
        }
        c[t] = 1 / c[t];
        for (int i = 0; i < N; i++) {
            alpha[t][i] = c[t] * alpha[t][i];
        }
    }
    /* This phase is the beta-pass. Similar to the alpha-pass, the values are scaled. */
    for (int i = 0; i < N; i++)
        beta[T - 1][i] = c[T - 1];
    for (int t = (T - 2); t >= 0; t--) {
        for (int i = 0; i < N; i++) {
            beta[t][i] = 0;
            for (int j = 0; j < N; j++) {
                beta[t][i] = beta[t][i] + A[i][j] * B[j][O[t + 1]] * beta[t + 1][j];
            }
            beta[t][i] = c[t] * beta[t][i];
        }
    }
    /* 
       The alpha and beta matrices are used to caculate the gamma and di-gamma matrices.
       Similar to the alpha and beta matrices, they are scaled. 
    */
    for (int t = 0; t < T - 1; t++) {
        float denom = 0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                denom = denom + alpha[t][i] * A[i][j] * B[j][O[t + 1]] * beta[t + 1][j];
            }
        }
        for (int i = 0; i < N; i++) {
            gamma[t][i] = 0;
            for (int j = 0; j < N; j++) {
                digamma[t][i][j] = (alpha[t][i] * A[i][j] * B[j][O[t + 1]] * beta[t + 1][j]) / denom;
                gamma[t][i] = gamma[t][i] + digamma[t][i][j];
            }
        }
    }
    /* 
     *************** Special case scenario for gamma ****************
     */
    float denom = 0;
    for (int i = 0; i < N; i++) {
        denom = denom + alpha[T - 1][i];
    }
    for (int i = 0; i < N; i++) {
        gamma[T - 1][i] = alpha[T - 1][i] / denom;
    }
    /* 
       Re-estimate the pi and B matrices based on the calculated gamma and di-gamma valaues.
       This is nothing but "training the model".
    */
    for (int i = 0; i < N; i++) {
        pi[i] = gamma[0][i];
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            float numer = 0;
            float denom = 0;
            for (int t = 0; t < T; t++) {
                if (O[t] == j) {
                    numer = numer + gamma[t][i];
                }
                denom = denom + gamma[t][i];
            }
            B[i][j] = numer / denom;
        }
    }
    /* 
       We need to computelog(P(O|λ)).
       This is used as part of the end conditions to stop training the model. 
    */
    float logProb = 0;
    for (int i = 0; i < T; i++) {
        logProb = logProb + log(c[i]);
    }
    logProb = -logProb;
    /* 
      Also we need to have a iterations check along with the probability calculated in order to decide
      when we need to stop training the model. 
      if (condition not met)
        continue algorithm
      else
        print the final A, B and pi matrices 
    */
    iters = iters + 1;
    if (iters < maxIterations || logProb > oldLogProb) {
        oldLogProb = logProb;
        goto alpha;
    } else {
        printf("\n");
        printf("A\n\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++)
                printf("%f\t", A[i][j]);
            printf("\n");
        }
        printf("\n\n");
        printf("B\n\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++)
                printf("%f\t", B[i][j]);
            printf("\n");
        }
        printf("\n\n");
        printf("Pi\n\n");
        for (int i = 0; i < N; i++) {
            printf("%f\t", pi[i]);
        }
        printf("\n\n");
    }
    /*
       This is the logic used to Calculate the number of correct positions (cipher to plain text mapping).
       We can thus use this information in order to claculate the putative key for the cipher text.
    */
    printf("\nPutative Key: \n");
    printf("Incorrect sequence values indicate incorrect plain-cipher text mapping\n");
    printf("K = %d\n", K);
    printf("Sequence based on key : Max Probability\n");
    for (int i = 0; i < N; i++) {
        printf("\n");
        double maxProbVal = 0;
        int charPosition;
        for (int j = 0; j < M; j++) {
            if (maxProbVal < B[i][j]) {
                maxProbVal = B[i][j];
                charPosition = j;
            }
        }
        printf("%d : %lf\n", charPosition, maxProbVal);
    }
    return 0;

}
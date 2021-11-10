/*************************
 ******* TRANSPOSE *******
 *************************/
template <class T>
void matTransposeKer(T* A, T* B, int heightA, int widthA) {
    for(int i = 0; i < heightA; i++){
        for (int j = 0; j < widthA; j++){
            B[j][i] = A[i][j];
        }
    }
}
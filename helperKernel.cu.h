#include <cuda_runtime.h>

/**
 * Generic Add operator that can be instantiated over
 *  numeric-basic types, such as int32_t, int64_t,
 *  float, double, etc.
 */
template<class T>
class Add {
  public:
    typedef T InpElTp;
    typedef T RedElTp;
    static const bool commutative = true;
    static __device__ __host__ inline T identInp()                    { return (T)0;    }
    //static __device__ __host__ inline T mapFun(const T& el)           { return el;      } // TODO: remove
    static __device__ __host__ inline T identity()                    { return (T)0;    }
    static __device__ __host__ inline T apply(const T t1, const T t2) { return t1 + t2; }

    //static __device__ __host__ inline bool equals(const T t1, const T t2) { return (t1 == t2); }
    static __device__ __host__ inline T remVolatile(volatile T& t)    { T res = t; return res; }
};

/***************************************************/
/*** Generic Value-Flag Tuple for Segmented Scan ***/
/***************************************************/

/**
 * Generic data-type that semantically tuples template
 * `T` with a flag, which is represented as a char.
 */
template<class T>
class ValFlg {
  public:
    T    v;
    char f;
    __device__ __host__ inline ValFlg() { f = 0; }
    __device__ __host__ inline ValFlg(const char& f1, const T& v1) { v = v1; f = f1; }
    __device__ __host__ inline ValFlg(const ValFlg& vf) { v = vf.v; f = vf.f; } 
    __device__ __host__ inline void operator=(const ValFlg& vf) volatile { v = vf.v; f = vf.f; }
};

/**
 * Generic segmented-scan operator, which lifts a generic
 * associative binary operator, given by generic type `OP`
 * to the corresponding segmented operator that works over
 * flag-value tuples.
 */
template<class OP>
class LiftOP {
  public:
    typedef ValFlg<typename OP::RedElTp> RedElTp;
    static __device__ __host__ inline RedElTp identity() {
        return RedElTp( (char)0, OP::identity());
    }

    static __device__ __host__ inline RedElTp
    apply(const RedElTp t1, const RedElTp t2) {
        typename OP::RedElTp v;
        char f = t1.f | t2.f;
        if (t2.f != 0) v = t2.v;
        else v = OP::apply(t1.v, t2.v);
        return RedElTp(f, v);
    }

    static __device__ __host__ inline bool
    equals(const RedElTp t1, const RedElTp t2) { 
        return ( (t1.f == t2.f) && OP::equals(t1.v, t2.v) ); 
    }
};

/*************************
 * TRANSPOSE
 * **********************/
template <class T>
__global__ void matTransposeKer(T* A, T* B, int heightA, int widthA) {
  int gidx = blockIdx.x*blockDim.x + threadIdx.x;
  int gidy = blockIdx.y*blockDim.y + threadIdx.y; 

  if( (gidx >= widthA) || (gidy >= heightA) ) return;

  B[gidx*heightA+gidy] = A[gidy*widthA + gidx];
}


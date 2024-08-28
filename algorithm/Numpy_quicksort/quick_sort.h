#ifndef __myself__sort__h__h__
#define __myself__sort__h__h__

#define PYA_QS_STACK 100
#define SMALL_QUICKSORT 15
#define SMALL_MERGESORT 20

#include <iostream>

template <typename T>
void TYPE_SWAP(T *a, T *b) {
    T t = *a;
    *a = *b;
    *b = t;
}

template <typename T>
bool TYPE_LT(T a, T b) {
    return a < b;
}

int npy_get_msb(int n) {
    int k;
    for (k = 0; n > 1; n >>= 1)
        ++k;
    return k;
}

template <typename T>
int heap_sort(T *start, int n) {
    T tmp, *a;
    int i, j, l;

    /* The array needs to be offset by one for heapsort indexing */
    a = start - 1;
    //先建立大顶堆
    for (l = n >> 1; l > 0; --l) {
        tmp = a[l];
        for (i = l, j = l << 1; j <= n;) {
            //因为假设根结点的序号是1，所以i结点左孩子和右孩子分别为2j和2i+1
            if (j < n && TYPE_LT(a[j], a[j + 1])) //左右孩子的比较
            {
                j += 1; //j为较大的记录的下标
            }
            if (TYPE_LT(tmp, a[j])) {
                //将孩子结点上位，则以孩子结点的位置进行下一轮的筛选
                a[i] = a[j];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        a[i] = tmp; //插入最开始不和谐的元素
    }
    //进行排序
    for (; n > 1;) {
        //最后一个元素和第一元素进行交换
        tmp = a[n];
        a[n] = a[1];
        n -= 1;
        //然后将剩下的无序元素继续调整为大顶堆
        for (i = 1, j = 2; j <= n;) {
            if (j < n && TYPE_LT(a[j], a[j + 1])) {
                j++;
            }
            if (TYPE_LT(tmp, a[j])) {
                a[i] = a[j];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        a[i] = tmp;
    }
    return 0;
}
template <typename T>
int aheapsort(T *vv, size_t *tosort, int n) {
    T *v = vv;
    size_t *a, i, j, l, tmp;
    /* The arrays need to be offset by one for heapsort indexing */
    a = tosort - 1;
    for (l = n >> 1; l > 0; --l) {
        tmp = a[l];
        for (i = l, j = l << 1; j <= n;) {
            if (j < n && TYPE_LT(v[a[j]], v[a[j + 1]])) {
                j += 1;
            }
            if (TYPE_LT(v[tmp], v[a[j]])) {
                a[i] = a[j];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        a[i] = tmp;
    }

    for (; n > 1;) {
        tmp = a[n];
        a[n] = a[1];
        n -= 1;
        for (i = 1, j = 2; j <= n;) {
            if (j < n && TYPE_LT(v[a[j]], v[a[j + 1]])) {
                j++;
            }
            if (TYPE_LT(v[tmp], v[a[j]])) {
                a[i] = a[j];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        a[i] = tmp;
    }

    return 0;
}

template <typename T>
int quick_sort(T *start, int num) {
    T vp;
    T *pl = start;
    T *pr = pl + num - 1;
    T *stack[PYA_QS_STACK];
    T **sptr = stack;
    T *pm, *pi, *pj, *pk;
    int depth[PYA_QS_STACK];
    int *psdepth = depth;
    int cdepth = npy_get_msb(num) * 2;

    for (;;) {
        if (cdepth < 0) {
            heap_sort(pl, pr - pl + 1);
            goto stack_pop;
        }
        while ((pr - pl) > SMALL_QUICKSORT) {
            /* quicksort partition */
            pm = pl + ((pr - pl) >> 1);
            if (TYPE_LT(*pm, *pl))
                TYPE_SWAP(pm, pl);
            if (TYPE_LT(*pr, *pm))
                TYPE_SWAP(pr, pm);
            if (TYPE_LT(*pm, *pl))
                TYPE_SWAP(pm, pl);
            vp = *pm;
            pi = pl;
            pj = pr - 1;
            TYPE_SWAP(pm, pj);
            for (;;) {
                do
                    ++pi;
                while (TYPE_LT(*pi, vp));
                do
                    --pj;
                while (TYPE_LT(vp, *pj));
                if (pi >= pj) {
                    break;
                }
                TYPE_SWAP(pi, pj);
            }
            pk = pr - 1;
            TYPE_SWAP(pi, pk);
            /* push largest partition on stack */
            if (pi - pl < pr - pi) {
                *sptr++ = pi + 1;
                *sptr++ = pr;
                pr = pi - 1;
            } else {
                *sptr++ = pl;
                *sptr++ = pi - 1;
                pl = pi + 1;
            }
            *psdepth++ = --cdepth;
        }

        /* insertion sort */
        for (pi = pl + 1; pi <= pr; ++pi) {
            vp = *pi;
            pj = pi;
            pk = pi - 1;
            while (pj > pl && TYPE_LT(vp, *pk)) {
                *pj-- = *pk--;
            }
            *pj = vp;
        }
    stack_pop:
        if (sptr == stack) {
            break;
        }
        pr = *(--sptr);
        pl = *(--sptr);
        cdepth = *(--psdepth);
    }
    return 0;
}

template <typename T>
int aquick_sort(T *start, size_t *tosort, int num) { // num 为size_t或者int 都可以
    T *v = start;
    T vp;
    size_t *pl = tosort;
    size_t *pr = tosort + num - 1;
    size_t *stack[PYA_QS_STACK];
    size_t **sptr = stack;
    size_t *pm, *pi, *pj, *pk, vi;
    int depth[PYA_QS_STACK];
    int *psdepth = depth;
    int cdepth = npy_get_msb(num) * 2;

    for (;;) {
        if (cdepth < 0) {
            aheapsort(start, pl, pr - pl + 1);
            goto stack_pop;
        }
        while ((pr - pl) > SMALL_QUICKSORT) {
            /* quicksort partition */
            pm = pl + ((pr - pl) >> 1);
            if (TYPE_LT(v[*pm], v[*pl]))
                std::swap(*pm, *pl);
            if (TYPE_LT(v[*pr], v[*pm]))
                std::swap(*pr, *pm); //std::swap(*pr, *pm);
            if (TYPE_LT(v[*pm], v[*pl]))
                std::swap(*pm, *pl);
            vp = v[*pm];
            pi = pl;
            pj = pr - 1;
            std::swap(*pm, *pj);
            for (;;) {
                do
                    ++pi;
                while (TYPE_LT(v[*pi], vp));
                do
                    --pj;
                while (TYPE_LT(vp, v[*pj]));
                if (pi >= pj) {
                    break;
                }
                std::swap(*pi, *pj);
            }
            pk = pr - 1;
            std::swap(*pi, *pk);
            /* push largest partition on stack */
            if (pi - pl < pr - pi) {
                *sptr++ = pi + 1;
                *sptr++ = pr;
                pr = pi - 1;
            } else {
                *sptr++ = pl;
                *sptr++ = pi - 1;
                pl = pi + 1;
            }
            *psdepth++ = --cdepth;
        }

        /* insertion sort */
        for (pi = pl + 1; pi <= pr; ++pi) {
            vi = *pi;
            vp = v[vi];
            pj = pi;
            pk = pi - 1;
            while (pj > pl && TYPE_LT(vp, v[*pk])) {
                *pj-- = *pk--;
            }
            *pj = vi;
        }
    stack_pop:
        if (sptr == stack) {
            break;
        }
        pr = *(--sptr);
        pl = *(--sptr);
        cdepth = *(--psdepth);
    }
    return 0;
}

#endif
// #define PYA_QS_STACK (sizeof(size_t) * 2)
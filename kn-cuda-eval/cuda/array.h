template<typename T, int R>
class Array {
public:
    Array() = default;

    Array(T *data) {
        memcpy(this->data, data, R * sizeof(T));
    }

    __host__ __device__

    T &operator[](int index) {
        return this->data[index];
    }

private:
    T data[R];
};

#include <stdio.h>
#include <iostream>
#include <fstream>

#ifndef INPUT_LEN

#define INPUT_LEN 100
#define TEST_LEN 250

#endif

using namespace std;

typedef struct Point_ {
    double x,y,z;
} Point;

typedef struct Input_{
    double camera_x;
    double camera_y;
    double camera_z;
    double line_rad_xy[INPUT_LEN];
    double line_rad_xz[INPUT_LEN];
    double timestamps[INPUT_LEN];
    int seq_len;
} Input; 

typedef struct Data_ {
    Input inputs[2];
    Point curvePoints[TEST_LEN];
    double curveTimestamps[TEST_LEN];
} Data;

typedef struct FileDataHeader_ {
    int data_length;
    Data* data;
} FileDataHeader;

extern "C" void* loadFromFile(const char* file_name) {
    FileDataHeader* file_data_header = new FileDataHeader;
    fstream fh;
    fh.open(file_name, ios::in | ios::binary);
    if (!fh.is_open()) {
        return nullptr;
    }
    fh.read(((char*)file_data_header), sizeof(FileDataHeader));

    Data* data = new Data[file_data_header->data_length];
    fh.read(((char*)data), sizeof(Data) * file_data_header->data_length);
    fh.close();

    file_data_header->data = data;
    return (void*)file_data_header;
}

extern "C"
bool loadIsSuccess(void* data) {
    return data != nullptr;
}

extern "C"
int getFileDataLength(void* data) {
    FileDataHeader* header = (FileDataHeader*)data;
    return header->data_length;
}
 
extern "C"
void* getFileData(void* data, int index) {
    FileDataHeader* header = (FileDataHeader*)data;
    return &(header->data[index]);
}

extern "C"
void releaseData(void* data) {
    FileDataHeader* header = (FileDataHeader*)data;
    delete[] header->data;
    delete header;
}

extern "C"
void createData_test(char* fileName) {
    FileDataHeader header;
    header.data_length = 10;
    header.data = new Data[header.data_length];
    for (int i = 0; i < header.data_length; i++) {
    }
    fstream fh;
    fh.open(fileName, ios::out | ios::binary);
    fh.write(((char*)&header), sizeof(FileDataHeader));
    fh.write(((char*)header.data), sizeof(Data) * header.data_length);
    fh.close();
}

extern "C"
void* createHeader(int data_length) {
    FileDataHeader* header = new FileDataHeader;
    header->data_length = data_length;
    header->data = new Data[data_length];
    return (void*)header;
}

extern "C"
bool putData(void* header, int i, Data data) {
    if (i >= getFileDataLength(header)) {
        return false;
    }
    FileDataHeader* file_header = (FileDataHeader*)header;
    file_header->data[i] = data;
    return true;
}

extern "C"
bool saveToFile(void* header, const char* file_name) {
    fstream fh;
    fh.open(file_name, ios::out | ios::binary);
    if (!fh.is_open()) {
        return false;
    }
    FileDataHeader* file_header = (FileDataHeader*)header;
    fh.write(((char*)file_header), sizeof(FileDataHeader));
    fh.write(((char*)file_header->data), sizeof(Data) * file_header->data_length);
    fh.close();
    return true;
}

extern "C"
Data getFileData_sync(char* file_name, int index) {
    ifstream fh;
    fh.open(file_name, ios::in | ios::binary);
    if (!fh.is_open()) {
        return Data();
    }
    fh.seekg(0, ios::beg);
    fh.seekg(index * sizeof(Data) + sizeof(FileDataHeader), ios::beg);
    Data* data = new Data;
    fh.read(((char*)data), sizeof(Data));
    fh.close();
    Data d = *data;
    delete data;
    return d;
}


extern "C"
int getFileDataLength_sync(char* file_name) {
    fstream fh;
    fh.open(file_name, ios::in | ios::binary);
    if (!fh.is_open()) {
        return -1;
    }
    FileDataHeader* header = new FileDataHeader;
    fh.read(((char*)header), sizeof(FileDataHeader));
    fh.close();
    return header->data_length;
}

extern "C"
void createEmptyFile_sync(char* file_name, int data_length) {
    FILE *fp = fopen(file_name, "w");
    // write header
    FileDataHeader header;
    header.data_length = data_length;
    fwrite(&header, sizeof(FileDataHeader), 1, fp);
    // write data
    Data data;
    for (int i = 0; i < data_length; i++) {
        fwrite(&data, sizeof(Data), 1, fp);
    }
    fclose(fp);
}


extern "C"
void putData_sync(char* file_name, int index, Data data) {
    FILE *fp = fopen(file_name, "r+");
    fseek(fp, index * sizeof(Data) + sizeof(FileDataHeader), SEEK_SET);
    fwrite(&data, sizeof(Data), 1, fp);
    fclose(fp);
}

extern "C"
void merge(char* a, char* b, char* new_file_name) {
    int a_l = getFileDataLength_sync(a);
    int b_l = getFileDataLength_sync(b);
    int new_l = a_l + b_l;
    createEmptyFile_sync(new_file_name, new_l);
    for (int i = 0; i < a_l; i++) {
        putData_sync(new_file_name, i, getFileData_sync(a, i));
    }
    for (int i = 0; i < b_l; i++) {
        putData_sync(new_file_name, i + a_l, getFileData_sync(b, i));
    }
}

extern "C"
int main() {
    std::cout << TEST_LEN << std::endl;
    return 0;
}
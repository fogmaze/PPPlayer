#include <stdio.h>
#include <iostream>
#include <fstream>
#define INPUT_LEN 20
#define TEST_LEN 200

using namespace std;

typedef struct Point_ {
    double x,y;
} Point;

typedef struct Input_{
    double camera_x;
    double camera_y;
    double camera_z;
    double line_deg_xy[INPUT_LEN];
    double line_deg_xz[INPUT_LEN];
    double timestamps[INPUT_LEN];
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
bool putData(void* header, int i) {
    if (i >= getFileDataLength(header)) {
        return false;
    }
    FileDataHeader* file_header = (FileDataHeader*)header;
    return true;
}

extern "C"
int main() {
    char const * file_name = "test.bin";
    createData_test((char*)file_name);
    FileDataHeader* header = (FileDataHeader*)loadFromFile("test.bin");
    for (int i = 0; i < header->data_length; i++) {
    }
    releaseData(header);
    return 0;
}




#include <GL/glut.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>

#include <vector>
#include <algorithm>
// 시간 측정등 고성능 함수
#include <chrono> 
#include <cuda_runtime.h>   // cudaMalloc 등 (커널은 cumian.cu)

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 800
#define WIDTH   512
#define HEIGHT  512
#define VOLX 256
#define VOLY 256	
#define VOLZ 225
const int BSIZE = 8;
//const int BSHIFT = 3;

//수정A-1: 전역 파라미터로 관리(BSIZE로 나누되, 나머지 있음 +1)
const int BZ_COUNT = VOLZ / BSIZE + (VOLZ % BSIZE != 0); // 29
const int BY_COUNT = VOLY / BSIZE + (VOLY % BSIZE != 0); // 32
const int BX_COUNT = VOLX / BSIZE + (VOLX % BSIZE != 0); // 32

const int ISO = 120; //iso


//unsigned char ImageBuf[HEIGHT][WIDTH];
unsigned char MyTexture[HEIGHT][WIDTH][3];
unsigned char vol[VOLZ][VOLY][VOLX];


// 제거 : bm(블록 최소값). 이진에서는 "1이 하나라도 있나"만 보면 되므로
unsigned char bM[BZ_COUNT][BY_COUNT][BX_COUNT];

using namespace std;


// 1단계 : 등방 PCA
const int   R_KER = 3;                  // 정육면체 창 반경
const float SIGMA = R_KER / 3.0f;       // 가우시안 폭 (3 sigma = R)
// 2단계 : 타원체 PCA
const float SIGMA_N = 0.6f;             // 법선 방향 폭
const float SIGMA_T_MAX = 4.0f;         // 표면 방향 최대 폭
const float RHO_FALLBACK = 0.7f;        // rho 가 이 이상이면 1차 공분산을 그대로 씀
const float RHO_SCALE = 0.3f;           // 미끄럼틀 s = 1/(1+(rho/RHO_SCALE)^2) 의 척도

// GPU 메모리
unsigned char* d_vol = 0;   // 이진 볼륨
unsigned char* d_bM = 0;    // 블록 최대값
float* d_A = 0;             // 1차 공분산 6성분 (xx, yy, zz, xy, xz, yz), 띠 밖 = 0
float* d_B = 0;             // 2차 공분산 6성분, 띠 밖 = 0
unsigned char* d_img = 0;   // 렌더 결과 RGB

// cumian.cu 의 창구 함수
void GpuStep1(const unsigned char* d_vol, float* d_A, int z, int R, float sigma);
void GpuStep2(const unsigned char* d_vol, const float* d_A, float* d_B, int z,
	float sn, float stMax, float rhoFallback, float rhoScale);
void GpuRender(const unsigned char* d_vol, const unsigned char* d_bM, const float* d_B, unsigned char* d_img,
	const float eye[3], const float u[3], const float v[3], const float w[3]);

//가벼운 함수----------------------------------------------------------
void FileRead()
{
	std::ifstream myfile;
	myfile.open("bighead.den", std::ios::in | std::ios::binary);
	if (!myfile.is_open()) {
		std::cout << "file error";
	}
	myfile.read((char*)vol, VOLZ * VOLY * VOLX);
	myfile.close();
}

void GenBlocks() { //수정A-2: 29, 32, 32에서 각각 B~_COUNT
	for (int bz = 0; bz < BZ_COUNT; bz++) // BZ = 28 
		for (int by = 0; by < BY_COUNT; by++)
			for (int bx = 0; bx < BX_COUNT; bx++) { // 각 블록에 대해서
				unsigned char max_value = 0;
				// 최대값을 추출해서 //(개선+; 경계값 추가)
				for (int z = bz * BSIZE; z <= __min(bz * BSIZE + BSIZE, VOLZ - 1); z++) { // 28*8 = for 224      z<232      vol[226]
					for (int y = by * BSIZE; y <= __min(by * BSIZE + BSIZE, VOLY - 1); y++) {
						for (int x = bx * BSIZE; x <= __min(bx * BSIZE + BSIZE, VOLX - 1); x++) { //bx=31, 31*8=248~256
							max_value = __max(vol[z][y][x], max_value);
						}
					}
				}
				// 저장한다.
				bM[bz][by][bx] = max_value;
			}
	printf("max = %d \n", bM[14][16][16]);
}

// 사진 저장 (BMP)
void SaveBMP(const char* filename) {
	int pad = (4 - (WIDTH * 3) % 4) % 4;
	int dataSize = (WIDTH * 3 + pad) * HEIGHT;
	int fileSize = 54 + dataSize;
	unsigned char header[54] = { 0 };
	header[0] = 'B'; header[1] = 'M';
	header[2] = fileSize; header[3] = fileSize >> 8;
	header[4] = fileSize >> 16; header[5] = fileSize >> 24;
	header[10] = 54; header[14] = 40;
	header[18] = WIDTH; header[19] = WIDTH >> 8;
	header[20] = WIDTH >> 16; header[21] = WIDTH >> 24;
	header[22] = HEIGHT; header[23] = HEIGHT >> 8;
	header[24] = HEIGHT >> 16; header[25] = HEIGHT >> 24;
	header[26] = 1; header[28] = 24;
	header[34] = dataSize; header[35] = dataSize >> 8;
	header[36] = dataSize >> 16; header[37] = dataSize >> 24;

	std::ofstream f(filename, std::ios::out | std::ios::binary);
	if (!f.is_open()) { std::cout << "save error : " << filename << std::endl; return; }
	f.write((char*)header, 54);
	unsigned char padding[3] = { 0, 0, 0 };
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
			unsigned char bgr[3] = { MyTexture[y][x][2], MyTexture[y][x][1], MyTexture[y][x][0] };
			f.write((char*)bgr, 3);
		}
		f.write((char*)padding, pad);
	}
	f.close();
	std::cout << "saved : " << filename << std::endl;
}

// CUDA 에러가 있으면 출력
void CudaCheck(cudaError_t err, const char* what) {
	if (err != cudaSuccess) std::cout << "[CUDA 에러] " << what << " : " << cudaGetErrorString(err) << std::endl;
}

void Render(glm::vec3 eye) {
	using namespace glm;

	glm::vec3 at(128, 128, 112);
	glm::vec3 up(0, 1, 0);

	glm::vec3 w = glm::normalize(at - eye);
	glm::vec3 u = glm::normalize(glm::cross(up, w));
	glm::vec3 v = glm::normalize(glm::cross(w, u));

	auto start = std::chrono::high_resolution_clock::now();
	int nPix = WIDTH * HEIGHT;
	float e3[3] = { eye.x, eye.y, eye.z }, u3[3] = { u.x, u.y, u.z }, v3[3] = { v.x, v.y, v.z }, w3[3] = { w.x, w.y, w.z };
	GpuRender(d_vol, d_bM, d_B, d_img, e3, u3, v3, w3);
	CudaCheck(cudaGetLastError(), "렌더 실행");
	CudaCheck(cudaDeviceSynchronize(), "렌더 수행");
	CudaCheck(cudaMemcpy(&MyTexture[0][0][0], d_img, nPix * 3, cudaMemcpyDeviceToHost), "이미지 복사");
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "실행 시간: " << duration.count() * 0.001f << " ms" << std::endl;
	glTexImage2D(GL_TEXTURE_2D, 0, 3, WIDTH, HEIGHT, 0, GL_RGB,
		GL_UNSIGNED_BYTE, &MyTexture[0][0][0]);
}

void MyInit() {
	glClearColor(0.0, 0.0, 0.0, 0.0);
	FileRead();


	// 이진화 전처리. 반드시 GenBlocks() 보다 먼저 와야 한다.
	for (int z = 0; z < VOLZ; z++)
		for (int y = 0; y < VOLY; y++)
			for (int x = 0; x < VOLX; x++)
				vol[z][y][x] = (vol[z][y][x] >= ISO);
	printf("binarized (ISO = %d, inside : d >= ISO)\n", ISO);
	//--------------------------------------------------------------------

	GenBlocks(); // 파일은 읽고 난 다음에.
	//==================== GPU 메모리 준비 ====================
	size_t nVox = (size_t)VOLX * VOLY * VOLZ;
	size_t covBytes = nVox * 6 * sizeof(float);
	CudaCheck(cudaMalloc(&d_vol, nVox), "d_vol 할당");
	CudaCheck(cudaMemcpy(d_vol, vol, nVox, cudaMemcpyHostToDevice), "d_vol 복사");
	CudaCheck(cudaMalloc(&d_bM, sizeof(bM)), "d_bM 할당");
	CudaCheck(cudaMemcpy(d_bM, bM, sizeof(bM), cudaMemcpyHostToDevice), "d_bM 복사");
	CudaCheck(cudaMalloc(&d_A, covBytes), "d_A 할당");
	CudaCheck(cudaMemset(d_A, 0, covBytes), "d_A 초기화");
	CudaCheck(cudaMalloc(&d_B, covBytes), "d_B 할당");
	CudaCheck(cudaMemset(d_B, 0, covBytes), "d_B 초기화");
	CudaCheck(cudaMalloc(&d_img, WIDTH * HEIGHT * 3), "d_img 할당");
	std::cout << "GPU 메모리 : A, B 각 " << covBytes / (1024 * 1024) << " MB" << std::endl;

	//==================== 1단계 : 등방 PCA -> A ====================
	auto pre1Start = std::chrono::high_resolution_clock::now();
	for (int z = 0; z < VOLZ; z++) {           // z 한 장씩 (한 번에 너무 오래 돌면 윈도우가 GPU를 리셋함)
		GpuStep1(d_vol, d_A, z, R_KER, SIGMA);
		CudaCheck(cudaGetLastError(), "1단계 실행");
	}
	CudaCheck(cudaDeviceSynchronize(), "1단계 수행");
	auto pre1End = std::chrono::high_resolution_clock::now();
	std::cout << "1단계 (R=" << R_KER << ") : "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(pre1End - pre1Start).count() << " ms" << std::endl;

	//==================== 2단계 : 타원체 PCA -> B ====================
	auto pre2Start = std::chrono::high_resolution_clock::now();
	for (int z = 0; z < VOLZ; z++) {
		GpuStep2(d_vol, d_A, d_B, z, SIGMA_N, SIGMA_T_MAX, RHO_FALLBACK, RHO_SCALE);
		CudaCheck(cudaGetLastError(), "2단계 실행");
		if (z % 10 == 0) {
			CudaCheck(cudaDeviceSynchronize(), "2단계 수행");
			std::cout << "  2단계 z = " << z << " / " << VOLZ << std::endl;
		}
	}
	CudaCheck(cudaDeviceSynchronize(), "2단계 수행");
	auto pre2End = std::chrono::high_resolution_clock::now();
	std::cout << "2단계 (sn=" << SIGMA_N << ", st최대=" << SIGMA_T_MAX << ", rho폴백=" << RHO_FALLBACK
		<< ", rho척도=" << RHO_SCALE << ") : "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(pre2End - pre2Start).count() << " ms" << std::endl;

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
	glEnable(GL_TEXTURE_2D);
}

void MyDisplay() {
	////////////////카메라 세팅
	glm::vec3 eye(0, 0, 100);   // 비교를 위해 고정
	cout << glm::to_string(eye) << endl;

	Render(eye);
	static int saved = 0;                       // 사진은 처음 한 번만 저장
	if (saved == 0) {
		SaveBMP("v0_코드정리.bmp");
		saved = 1;
	}
	glClear(GL_COLOR_BUFFER_BIT);
	glBegin(GL_QUADS);
	float fSize = 0.8f;
	glTexCoord2f(0.0, 0.0); glVertex3f(-fSize, -fSize, 0.0);
	glTexCoord2f(0.0, 1.0); glVertex3f(-fSize, fSize, 0.0);
	glTexCoord2f(1.0, 1.0); glVertex3f(fSize, fSize, 0.0);
	glTexCoord2f(1.0, 0.0); glVertex3f(fSize, -fSize, 0.0);
	glEnd();
	glutSwapBuffers();
}

int main(int argc, char** argv) {
	glutInit(&argc, argv); //GLUT 윈도우 함수
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutCreateWindow("OpenGL Drawing Example");
	MyInit();
	glutDisplayFunc(MyDisplay);
	//glutIdleFunc(MyDisplay);
	glutMainLoop();   // 창을 띄워 MyDisplay(렌더+저장)가 돌도록
	return 0;
}
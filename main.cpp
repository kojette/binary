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
#define WIDTH   1024        // (2차_4차) 512 -> 1024 : 512 데이터 폭 전체가 화면에 들어오도록
#define HEIGHT  1024
const int BSIZE = 8;
//const int BSHIFT = 3;

//==================== (2차_4차) 데이터별 전역 파라미터 : 1 -> 2 -> 3 순서로 돈다 ====================
const int   FILE_COUNT = 3;
const char* FILE_NAME[FILE_COUNT] = { "bighead.den", "abdomen300.den", "output_HEAD_01" };
const int   FILE_TYPE[FILE_COUNT] = { 0, 1, 2 };            // 0 = unsigned char, 1 = unsigned short, 2 = signed short (모두 리틀 엔디언, 분석 로그로 확인)
const int   FILE_X[FILE_COUNT] = { 256, 512, 512 };
const int   FILE_Y[FILE_COUNT] = { 256, 512, 512 };
const int   FILE_Z[FILE_COUNT] = { 225, 300, 552 };      // 파일 크기 ÷ (512*512*2) 로 확인한 장 수
// 이진화 문턱 : 파일값 >= ISO_HU + HU_OFS 이면 안쪽
//  1 bighead : 8비트라 HU 아님 -> 원래 값 120 그대로 (연조직 봉우리 88 과 뼈 봉우리 165 사이)
//  2 abdomen : 파일값 = HU + 1024,  3 HEAD_01 : 파일값 = HU  (분석 로그 [B][E] 로 추정)
//  뼈 문턱 +300 HU 는 관례값 (2·3 은 히스토그램에 뼈 봉우리가 없어 골짜기를 못 찾음)
const int   FILE_ISO_HU[FILE_COUNT] = { 120, 350, 350 };
const int   FILE_HU_OFS[FILE_COUNT] = { 0, 1024, 0 };
const char* OUT_NAME = "(data+)_iso_120_350_350";   // 저장 이름 : OUT_NAME_1.bmp, OUT_NAME_2.bmp, OUT_NAME_3.bmp
// 시점 (파일별) : eye = at(볼륨 중심) + EYE_OFS, 화면 위쪽 = UP, 픽셀 하나 = PIXEL 칸
//  1 bighead : 기존 시점 그대로 (비교용)
//  2·3       : 정면 (몸 앞 = y 작은 쪽에서 +y 로 바라봄, 머리 = z 0 쪽이 화면 위)
const glm::vec3 FILE_EYE_OFS[FILE_COUNT] = { glm::vec3(-128, -128, -12), glm::vec3(0, -400, 0), glm::vec3(0, +400, 0) };
const glm::vec3 FILE_UP[FILE_COUNT] = { glm::vec3(0, 1, 0),         glm::vec3(0, 0, 1),   glm::vec3(0, 0, -1) };
const float     FILE_PIXEL[FILE_COUNT] = { 0.5f, 0.5f, 0.6f };   // 1024 x PIXEL = 화면 폭(칸). 3번은 552장이 들어오도록 0.6
int CUR = 0;                                                       // 지금 도는 데이터 번호 (MyInit 에서 채움)

// 지금 도는 데이터의 크기·문턱 (MyInit 에서 채움)
int VOLX, VOLY, VOLZ;
int BZ_COUNT, BY_COUNT, BX_COUNT;   // BSIZE로 나누되, 나머지 있음 +1
int ISO;


//unsigned char ImageBuf[HEIGHT][WIDTH];
unsigned char MyTexture[HEIGHT][WIDTH][3];
unsigned char* vol = 0;   // [VOLZ][VOLY][VOLX] 를 한 줄로 : vol[(z * VOLY + y) * VOLX + x]


// 제거 : bm(블록 최소값). 이진에서는 "1이 하나라도 있나"만 보면 되므로
unsigned char* bM = 0;    // [BZ_COUNT][BY_COUNT][BX_COUNT] 를 한 줄로

using namespace std;


// 1단계 : 등방 PCA
const int   R_KER = 3;                  // 정육면체 창 반경
const float SIGMA = R_KER / 3.0f;       // 가우시안 폭 (3 sigma = R)
// 2단계 : 타원체 PCA
const float SIGMA_N = 0.6f;             // 법선 방향 폭
const float SIGMA_T_MAX = 4.0f;         // 표면 방향 최대 폭

//반복pca
const float RHO_ROUND = 0.3f;   // 1차 λ3/λ2 가 이 이상이면 2차 1번째를 둥근 커널로
const int   ITER2 = 3;          // 2차 PCA 횟수 (1 = 지금과 같음)

// GPU 메모리
unsigned char* d_vol = 0;   // 이진 볼륨
unsigned char* d_bM = 0;    // 블록 최대값
float* d_A = 0;             // 1차 공분산 6성분 (xx, yy, zz, xy, xz, yz), 띠 밖 = 0
float* d_B = 0;             // 2차 공분산 6성분, 띠 밖 = 0
unsigned char* d_img = 0;   // 렌더 결과 RGB

// cumian.cu 의 창구 함수
void GpuSetDim(int volx, int voly, int volz, int byCount, int bxCount);   // (2차_4차) 볼륨 크기를 GPU 상수로
void GpuStep1(const unsigned char* d_vol, float* d_A, int z, int R, float sigma);
void GpuStep2(const unsigned char* d_vol, const float* d_A, float* d_B, int z,
	float sn, float stMax, float rhoRound, int iter2);
void GpuRender(const unsigned char* d_vol, const unsigned char* d_bM, const float* d_B, unsigned char* d_img,
	const float eye[3], const float u[3], const float v[3], const float w[3], float pixel);

//가벼운 함수----------------------------------------------------------
// (2차_4차) 파일 f 를 한 장(z)씩 읽으면서 바로 문턱 처리 -> vol 에는 0/1 unsigned char 만 저장
void FileRead(int f)
{
	std::ifstream myfile;
	myfile.open(FILE_NAME[f], std::ios::in | std::ios::binary);
	if (!myfile.is_open()) {
		std::cout << "file error : " << FILE_NAME[f] << std::endl;
	}
	int bytes = (FILE_TYPE[f] == 0) ? 1 : 2;
	size_t nSlice = (size_t)VOLX * VOLY;
	std::vector<unsigned char> buf(nSlice * bytes);   // 한 장 버퍼 (512*512*2 = 0.5MB)
	long long total = 0;
	for (int z = 0; z < VOLZ; z++) {
		myfile.read((char*)buf.data(), nSlice * bytes);
		total += myfile.gcount();
		for (size_t k = 0; k < nSlice; k++) {   // 파일별 캐스팅
			int d;
			if (FILE_TYPE[f] == 0)      d = buf[k];
			else if (FILE_TYPE[f] == 1) d = ((unsigned short*)buf.data())[k];
			else                        d = ((short*)buf.data())[k];
			vol[z * nSlice + k] = (d >= ISO);
		}
	}
	std::cout << "  읽은 바이트 " << total << " / 필요 " << (long long)nSlice * VOLZ * bytes << std::endl;   // [로그]
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
							max_value = __max(vol[((size_t)z * VOLY + y) * VOLX + x], max_value);
						}
					}
				}
				// 저장한다.
				bM[((size_t)bz * BY_COUNT + by) * BX_COUNT + bx] = max_value;
			}
	printf("max = %d \n", bM[(14 * BY_COUNT + 16) * BX_COUNT + 16]);
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

void Render() {
	using namespace glm;

	glm::vec3 at(VOLX / 2, VOLY / 2, VOLZ / 2);   // (2차_4차) 볼륨 중심. bighead 는 기존과 같은 (128, 128, 112)
	glm::vec3 eye = at + FILE_EYE_OFS[CUR];         // bighead 는 기존과 같은 (0, 0, 100)
	glm::vec3 up = FILE_UP[CUR];
	cout << glm::to_string(eye) << endl;

	glm::vec3 w = glm::normalize(at - eye);
	glm::vec3 u = glm::normalize(glm::cross(up, w));
	glm::vec3 v = glm::normalize(glm::cross(w, u));

	auto start = std::chrono::high_resolution_clock::now();
	int nPix = WIDTH * HEIGHT;
	float e3[3] = { eye.x, eye.y, eye.z }, u3[3] = { u.x, u.y, u.z }, v3[3] = { v.x, v.y, v.z }, w3[3] = { w.x, w.y, w.z };
	GpuRender(d_vol, d_bM, d_B, d_img, e3, u3, v3, w3, FILE_PIXEL[CUR]);
	CudaCheck(cudaGetLastError(), "렌더 실행");
	CudaCheck(cudaDeviceSynchronize(), "렌더 수행");
	CudaCheck(cudaMemcpy(&MyTexture[0][0][0], d_img, nPix * 3, cudaMemcpyDeviceToHost), "이미지 복사");
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "실행 시간: " << duration.count() * 0.001f << " ms" << std::endl;
	glTexImage2D(GL_TEXTURE_2D, 0, 3, WIDTH, HEIGHT, 0, GL_RGB,
		GL_UNSIGNED_BYTE, &MyTexture[0][0][0]);
}

// (2차_4차) 초기화 : 데이터 f 의 크기 설정 -> CPU 할당 -> 읽기·캐스팅·이진화 -> 블록 -> GPU 할당·복사
void MyInit(int f) {
	CUR = f;
	VOLX = FILE_X[f]; VOLY = FILE_Y[f]; VOLZ = FILE_Z[f];
	ISO = FILE_ISO_HU[f] + FILE_HU_OFS[f];   // 파일값 단위 문턱
	BZ_COUNT = VOLZ / BSIZE + (VOLZ % BSIZE != 0);
	BY_COUNT = VOLY / BSIZE + (VOLY % BSIZE != 0);
	BX_COUNT = VOLX / BSIZE + (VOLX % BSIZE != 0);
	size_t nVox = (size_t)VOLX * VOLY * VOLZ;
	size_t nBlk = (size_t)BZ_COUNT * BY_COUNT * BX_COUNT;
	const char* TYPE_NAME[3] = { "unsigned char", "unsigned short", "signed short" };
	printf("\n========== 데이터 %d : %s (%d x %d x %d, %s, ISO_HU = %d, HU_OFS = %d -> 파일값 ISO = %d) ==========\n",   // [로그]
		f + 1, FILE_NAME[f], VOLX, VOLY, VOLZ, TYPE_NAME[FILE_TYPE[f]], FILE_ISO_HU[f], FILE_HU_OFS[f], ISO);
	printf("  블록 %d x %d x %d\n", BX_COUNT, BY_COUNT, BZ_COUNT);                             // [로그]

	//CPU 메모리 : 이진 볼륨(0/1)과 블록만. 원본은 FileRead 에서 한 장씩만 잡았다 버림
	vol = (unsigned char*)malloc(nVox);
	bM = (unsigned char*)malloc(nBlk);
	FileRead(f);   // 읽기 + 캐스팅 + 이진화
	printf("binarized (ISO = %d, inside : d >= ISO)\n", ISO);

	GenBlocks(); // 블록별 최대값

	//GPU 메모리 준비
	size_t covBytes = nVox * 6 * sizeof(float);
	GpuSetDim(VOLX, VOLY, VOLZ, BY_COUNT, BX_COUNT);
	CudaCheck(cudaGetLastError(), "크기 설정");
	CudaCheck(cudaMalloc(&d_vol, nVox), "d_vol 할당");
	CudaCheck(cudaMemcpy(d_vol, vol, nVox, cudaMemcpyHostToDevice), "d_vol 복사");
	CudaCheck(cudaMalloc(&d_bM, nBlk), "d_bM 할당");
	CudaCheck(cudaMemcpy(d_bM, bM, nBlk, cudaMemcpyHostToDevice), "d_bM 복사");
	CudaCheck(cudaMalloc(&d_A, covBytes), "d_A 할당");
	CudaCheck(cudaMemset(d_A, 0, covBytes), "d_A 초기화");
	CudaCheck(cudaMalloc(&d_B, covBytes), "d_B 할당");
	CudaCheck(cudaMemset(d_B, 0, covBytes), "d_B 초기화");
	CudaCheck(cudaMalloc(&d_img, WIDTH * HEIGHT * 3), "d_img 할당");
	std::cout << "GPU 메모리 : A, B 각 " << covBytes / (1024 * 1024) << " MB" << std::endl;
}

// (2차_4차) 해제 : MyInit 에서 잡은 것 전부
void MyFree() {
	CudaCheck(cudaFree(d_vol), "d_vol 해제");
	CudaCheck(cudaFree(d_bM), "d_bM 해제");
	CudaCheck(cudaFree(d_A), "d_A 해제");
	CudaCheck(cudaFree(d_B), "d_B 해제");
	CudaCheck(cudaFree(d_img), "d_img 해제");
	d_vol = 0; d_bM = 0; d_A = 0; d_B = 0; d_img = 0;
	free(vol); free(bM);
	vol = 0; bM = 0;
	std::cout << "해제 완료" << std::endl;   // [로그]
}

// 마지막 데이터의 사진을 창에 띄우기만 함 (렌더·저장은 main 반복에서)
void MyDisplay() {
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
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
	glEnable(GL_TEXTURE_2D);
	printf("파라미터 : R = %d, sigma = %.2f, sn = %.2f, st = %.2f, rhoRound = %.2f, iter2 = %d\n",   // [로그]
		R_KER, SIGMA, SIGMA_N, SIGMA_T_MAX, RHO_ROUND, ITER2);

	for (int f = 0; f < FILE_COUNT; f++) {
		MyInit(f);

		//==================== 1단계 : 등방 PCA -> 1차 공분산 ====================
		auto pre1Start = std::chrono::high_resolution_clock::now();
		for (int z = 0; z < VOLZ; z++) {           // xy면 한 장씩.
			GpuStep1(d_vol, d_A, z, R_KER, SIGMA);
			CudaCheck(cudaGetLastError(), "1단계 실행");
		}
		CudaCheck(cudaDeviceSynchronize(), "1단계 수행");
		auto pre1End = std::chrono::high_resolution_clock::now();
		std::cout << "1단계 (R=" << R_KER << ") : "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(pre1End - pre1Start).count() << " ms" << std::endl;

		//==================== 2단계 : 타원체 PCA -> 2차 공분산 ====================
		auto pre2Start = std::chrono::high_resolution_clock::now();
		for (int z = 0; z < VOLZ; z++) {
			GpuStep2(d_vol, d_A, d_B, z, SIGMA_N, SIGMA_T_MAX, RHO_ROUND, ITER2);
			CudaCheck(cudaGetLastError(), "2단계 실행");
			if (z % 10 == 0) {
				CudaCheck(cudaDeviceSynchronize(), "2단계 수행");
				std::cout << "  2단계 z = " << z << " / " << VOLZ << std::endl;
			}
		}
		CudaCheck(cudaDeviceSynchronize(), "2단계 수행");
		auto pre2End = std::chrono::high_resolution_clock::now();
		std::cout << "2단계 (sn=" << SIGMA_N << ", st=" << SIGMA_T_MAX << ") : "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(pre2End - pre2Start).count() << " ms" << std::endl;

		//==================== 렌더 + 저장 ====================
		Render();
		char outName[256]; 
		snprintf(outName, sizeof(outName), "%s_%d.bmp", OUT_NAME, f + 1);
		SaveBMP(outName);

		MyFree();
	}

	glutDisplayFunc(MyDisplay);
	//glutIdleFunc(MyDisplay);
	glutMainLoop();   // 창에는 마지막 데이터 사진
	return 0;
}
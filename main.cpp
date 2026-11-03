// ============================================================================
// exp_main.cpp : 국내논문 실험 자료 일괄 생성 (일회성 실험 코드)
//  기준 : base 브랜치 main.cpp (v0_베이스확정)
//  바뀐 것
//   - 실험 A (판 비교, iso 고정) : 판 4개 x 시점(1번 1개, 2·3번 정면 + 45도)
//       1sphere : 2단계 생략, 렌더가 1차 공분산 배열을 읽음
//       2aniso  : 2단계 비등방 1회, 섞기 없음 (rhoLo 2.0, rhoHi 3.0 -> 섞는 비율 항상 0)
//       3base   : 2단계 비등방 1회 + 섞기 0.25~0.40 (= 기존 코드 그대로)
//       4normal : 3base 의 2차 공분산 -> 최소 고유벡터(법선 3 float) 저장, 렌더는 8꼭짓점 법선을
//                 시선 쪽으로 뒤집은 뒤 삼선형 보간 -> 정규화
//     쌍(사슬) : p1stage2 (1->2), p2mix (2->3), p3store (3->4). 쌍마다 바뀐 픽셀 % + 차이 x6 영상
//     3base 는 기존 사진 (base)v0_120_250_250_N.bmp 와 픽셀 비교 (기존 시점·정면만)
//   - 실험 B (iso 따로) : 2·3번, base 판, 정면만. A 와 같은 iso 는 A 사진 재사용
//   - iso 는 파일값 그대로 (HU 환산 HU_OFS 제거)
//   - 로그 : log_run.csv, log_band.csv, log_pair.csv, log_ref.csv + 화면
//  안 바뀐 것 : FileRead, GenBlocks, K_Step1, K_Step2, K_Render (base 판이 기존과 같은 결과를 내도록)
// ============================================================================
#define _CRT_SECURE_NO_WARNINGS   // (실험) fopen 허용 (MSVC 경고 C4996 끔)
#include <GL/glut.h>
#include <GL/glut.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>   // (실험) memcpy
#include <math.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>

#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>   // cudaMalloc 등 (커널은 cumian.cu)

#define WINDOW_WIDTH 1024
#define WINDOW_HEIGHT 1024
#define WIDTH   1024
#define HEIGHT  1024
const int BSIZE = 8;

//==================== 데이터별 전역 파라미터 ====================
const int   FILE_COUNT = 3;
const char* FILE_NAME[FILE_COUNT] = { "bighead.den", "abdomen300.den", "output_HEAD_01" };
const int   FILE_TYPE[FILE_COUNT] = { 0, 1, 2 };            // 0 = unsigned char, 1 = unsigned short, 2 = signed short
const int   FILE_X[FILE_COUNT] = { 256, 512, 512 };
const int   FILE_Y[FILE_COUNT] = { 256, 512, 512 };
const int   FILE_Z[FILE_COUNT] = { 225, 300, 552 };

// (실험) iso 는 파일값 그대로 : 파일값 >= iso 이면 안쪽
//  1 bighead : 8비트 120,  2 abdomen : 1274 (기존 뼈 기준과 같은 값),  3 HEAD_01 : signed 250
const int   A_ISO[FILE_COUNT] = { 120, 1274, 250 };
// (실험 B) iso 후보 : 피부 3개 + 뼈 3개. 1번은 하지 않음
const int   B_DO[FILE_COUNT] = { 0, 1, 1 };
const int   B_ISO_COUNT = 6;
const int   B_ISO[FILE_COUNT][B_ISO_COUNT] = {
	{ 0, 0, 0, 0, 0, 0 },
	{ 324, 524, 724, 1274, 1324, 1374 },
	{ -700, -500, -300, 250, 300, 350 } };

// 시점 (파일별) : eye = at(볼륨 중심) + EYE_OFS, 화면 위쪽 = UP, 픽셀 하나 = PIXEL 칸
const glm::vec3 FILE_EYE_OFS[FILE_COUNT] = { glm::vec3(-128, -128, -12), glm::vec3(0, -400, 0), glm::vec3(0, +400, 0) };
const glm::vec3 FILE_UP[FILE_COUNT] = { glm::vec3(0, 1, 0),         glm::vec3(0, 0, 1),   glm::vec3(0, 0, -1) };
const float     FILE_PIXEL[FILE_COUNT] = { 0.5f, 0.5f, 0.6f };
// (실험) 시점 : 0 = 기존(정면), 1 = 화면 위쪽 축 기준 +45도 회전. 1번은 기존 시점 하나만
const int   VIEW_COUNT[FILE_COUNT] = { 1, 2, 2 };
const char* VIEW_NAME[FILE_COUNT][2] = { { "orig", "" }, { "front", "rot45" }, { "front", "rot45" } };
const float VIEW_DEG[2] = { 0.0f, 45.0f };

// (실험) 판 이름과 쌍 이름 (쌍 k = 판 k -> 판 k+1)
const int   VAR_COUNT = 4;
const char* VAR_NAME[VAR_COUNT] = { "1sphere", "2aniso", "3base", "4normal" };
const char* PAIR_NAME[VAR_COUNT - 1] = { "p1stage2", "p2mix", "p3store" };
// (실험) base 재현 확인용 기존 사진 : REF_NAME_1.bmp ~ _3.bmp (실행 폴더에 있을 때만 비교)
const char* REF_NAME = "(base)v0_120_250_250";

int CUR = 0;
int VOLX, VOLY, VOLZ;
int BZ_COUNT, BY_COUNT, BX_COUNT;
int ISO;
size_t COV_BYTES = 0;   // (실험) 공분산 배열 하나의 바이트

unsigned char MyTexture[HEIGHT][WIDTH][3];
unsigned char* vol = 0;
unsigned char* bM = 0;

using namespace std;

// 1단계 : 등방 PCA
const int   R_KER = 3;
const float SIGMA = R_KER / 3.0f;
// 2단계 : 타원체 PCA (1회)
const float SIGMA_N = 0.6f;
const float SIGMA_T = 4.0f;
const float SIGMA_R = 3.0f;
const float RHO_LO = 0.25f;
const float RHO_HI = 0.40f;
// (실험) 2aniso 판 : 섞는 비율이 항상 0 이 되도록 (rho 는 최대 1 이므로 2.0 아래에서 t < 0 -> s = 0)
const float NOMIX_LO = 2.0f;
const float NOMIX_HI = 3.0f;

// GPU 메모리
unsigned char* d_vol = 0;
unsigned char* d_bM = 0;
float* d_A = 0;             // 1차 공분산 6성분
float* d_B = 0;             // 2차 공분산 6성분
float* d_N = 0;             // (실험) 4normal 판의 법선 3성분
unsigned char* d_img = 0;

// cumian.cu 의 창구 함수
void GpuSetDim(int volx, int voly, int volz, int byCount, int bxCount);
void GpuStep1(const unsigned char* d_vol, float* d_A, int z, int R, float sigma);
void GpuStep2(const unsigned char* d_vol, const float* d_A, float* d_B, int z,
	float sn, float st, float sr, float rhoLo, float rhoHi);
void GpuRender(const unsigned char* d_vol, const unsigned char* d_bM, const float* d_B, unsigned char* d_img,
	const float eye[3], const float u[3], const float v[3], const float w[3], float pixel);
// (실험) 추가 창구
void GpuStats(const float* d_A, unsigned long long cnt[4], float lo, float hi);
void GpuToNormal(const float* d_B, float* d_N);
void GpuRenderN(const unsigned char* d_vol, const unsigned char* d_bM, const float* d_N, unsigned char* d_img,
	const float eye[3], const float u[3], const float v[3], const float w[3], float pixel);

//가벼운 함수----------------------------------------------------------
void FileRead(int f)
{
	std::ifstream myfile;
	myfile.open(FILE_NAME[f], std::ios::in | std::ios::binary);
	if (!myfile.is_open()) {
		std::cout << "file error : " << FILE_NAME[f] << std::endl;
	}
	int bytes = (FILE_TYPE[f] == 0) ? 1 : 2;
	size_t nSlice = (size_t)VOLX * VOLY;
	std::vector<unsigned char> buf(nSlice * bytes);
	long long total = 0;
	for (int z = 0; z < VOLZ; z++) {
		myfile.read((char*)buf.data(), nSlice * bytes);
		total += myfile.gcount();
		for (size_t k = 0; k < nSlice; k++) {
			int d;
			if (FILE_TYPE[f] == 0)      d = buf[k];
			else if (FILE_TYPE[f] == 1) d = ((unsigned short*)buf.data())[k];
			else                        d = ((short*)buf.data())[k];
			vol[z * nSlice + k] = (d >= ISO);
		}
	}
	std::cout << "  읽은 바이트 " << total << " / 필요 " << (long long)nSlice * VOLZ * bytes << std::endl;
	myfile.close();
}

void GenBlocks() {
	for (int bz = 0; bz < BZ_COUNT; bz++)
		for (int by = 0; by < BY_COUNT; by++)
			for (int bx = 0; bx < BX_COUNT; bx++) {
				unsigned char max_value = 0;
				for (int z = bz * BSIZE; z <= __min(bz * BSIZE + BSIZE, VOLZ - 1); z++) {
					for (int y = by * BSIZE; y <= __min(by * BSIZE + BSIZE, VOLY - 1); y++) {
						for (int x = bx * BSIZE; x <= __min(bx * BSIZE + BSIZE, VOLX - 1); x++) {
							max_value = __max(vol[((size_t)z * VOLY + y) * VOLX + x], max_value);
						}
					}
				}
				bM[((size_t)bz * BY_COUNT + by) * BX_COUNT + bx] = max_value;
			}
	printf("max = %d \n", bM[(14 * BY_COUNT + 16) * BX_COUNT + 16]);
}

// (실험) 사진 저장 (BMP) : 버퍼 rgb[(y * WIDTH + x) * 3 + c] 를 저장 (기존 SaveBMP 와 같은 순서)
void SaveBMP(const char* filename, const unsigned char* rgb) {
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
			const unsigned char* p = rgb + ((size_t)y * WIDTH + x) * 3;
			unsigned char bgr[3] = { p[2], p[1], p[0] };
			f.write((char*)bgr, 3);
		}
		f.write((char*)padding, pad);
	}
	f.close();
	std::cout << "saved : " << filename << std::endl;
}

// (실험) 기존 사진과 픽셀 비교 : -1 = 파일 없음, -2 = 크기 다름, 0 이상 = 다른 픽셀 수
long long CompareWithFile(const char* filename, const unsigned char* rgb) {
	std::ifstream f(filename, std::ios::in | std::ios::binary);
	if (!f.is_open()) return -1;
	unsigned char header[54];
	f.read((char*)header, 54);
	int w = header[18] | (header[19] << 8) | (header[20] << 16) | (header[21] << 24);
	int h = header[22] | (header[23] << 8) | (header[24] << 16) | (header[25] << 24);
	if (w != WIDTH || h != HEIGHT) return -2;
	int pad = (4 - (WIDTH * 3) % 4) % 4;
	long long diff = 0;
	unsigned char bgr[3], padding[3];
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
			f.read((char*)bgr, 3);
			const unsigned char* p = rgb + ((size_t)y * WIDTH + x) * 3;
			if (bgr[0] != p[2] || bgr[1] != p[1] || bgr[2] != p[0]) diff++;
		}
		f.read((char*)padding, pad);
	}
	return diff;
}

// (실험) 쌍 비교 : 밝기 = RGB 평균, 바뀐 픽셀 = 밝기 차 > 8, 분모 = 둘 중 하나라도 배경(0,0,0)이 아닌 픽셀
//  차이 영상 : 흰 바탕, 밝기 차 x 6 만큼 어둡게 (진할수록 큼)
//  silhouette = 한쪽만 배경인 픽셀 (표면이 생기거나 사라진 곳)
void ComparePair(const unsigned char* a, const unsigned char* b, const char* diffName,
	long long& changed, long long& surf, long long& sil) {
	std::vector<unsigned char> d((size_t)WIDTH * HEIGHT * 3);
	changed = 0; surf = 0; sil = 0;
	for (size_t k = 0; k < (size_t)WIDTH * HEIGHT; k++) {
		const unsigned char* pa = a + k * 3;
		const unsigned char* pb = b + k * 3;
		int bgA = (pa[0] == 0 && pa[1] == 0 && pa[2] == 0);
		int bgB = (pb[0] == 0 && pb[1] == 0 && pb[2] == 0);
		float la = (pa[0] + pa[1] + pa[2]) / 3.0f;
		float lb = (pb[0] + pb[1] + pb[2]) / 3.0f;
		float dl = fabsf(la - lb);
		if (!(bgA && bgB)) surf++;
		if (bgA != bgB) sil++;
		if (!(bgA && bgB) && dl > 8.0f) changed++;
		int g = 255 - (int)fminf(255.0f, dl * 6.0f);
		d[k * 3 + 0] = d[k * 3 + 1] = d[k * 3 + 2] = (unsigned char)g;
	}
	SaveBMP(diffName, d.data());
}

void CudaCheck(cudaError_t err, const char* what) {
	if (err != cudaSuccess) std::cout << "[CUDA 에러] " << what << " : " << cudaGetErrorString(err) << std::endl;
}

// (실험) 벡터 v 를 단위축 k 둘레로 deg 도 회전 (로드리게스 공식)
glm::vec3 RotateAbout(glm::vec3 v, glm::vec3 k, float deg) {
	float a = deg * 3.14159265f / 180.0f;
	return v * cosf(a) + glm::cross(k, v) * sinf(a) + k * glm::dot(k, v) * (1.0f - cosf(a));
}

// (실험) 렌더 : view 0 = 기존 시점, 1 = 45도. arr = 렌더가 읽을 배열, useN = 1 이면 법선 배열 렌더
//  view 0 의 계산은 기존 Render 와 같음. 반환 = ms (기존처럼 이미지 복사까지 포함)
double Render(int view, const float* arr, int useN) {
	using namespace glm;

	glm::vec3 at(VOLX / 2, VOLY / 2, VOLZ / 2);
	glm::vec3 ofs = FILE_EYE_OFS[CUR];
	if (view != 0) ofs = RotateAbout(ofs, glm::normalize(FILE_UP[CUR]), VIEW_DEG[view]);
	glm::vec3 eye = at + ofs;
	glm::vec3 up = FILE_UP[CUR];
	cout << glm::to_string(eye) << endl;

	glm::vec3 w = glm::normalize(at - eye);
	glm::vec3 u = glm::normalize(glm::cross(up, w));
	glm::vec3 v = glm::normalize(glm::cross(w, u));

	auto start = std::chrono::high_resolution_clock::now();
	int nPix = WIDTH * HEIGHT;
	float e3[3] = { eye.x, eye.y, eye.z }, u3[3] = { u.x, u.y, u.z }, v3[3] = { v.x, v.y, v.z }, w3[3] = { w.x, w.y, w.z };
	if (useN == 0) GpuRender(d_vol, d_bM, arr, d_img, e3, u3, v3, w3, FILE_PIXEL[CUR]);
	else           GpuRenderN(d_vol, d_bM, arr, d_img, e3, u3, v3, w3, FILE_PIXEL[CUR]);
	CudaCheck(cudaGetLastError(), "렌더 실행");
	CudaCheck(cudaDeviceSynchronize(), "렌더 수행");
	CudaCheck(cudaMemcpy(&MyTexture[0][0][0], d_img, nPix * 3, cudaMemcpyDeviceToHost), "이미지 복사");
	auto end = std::chrono::high_resolution_clock::now();
	double ms = std::chrono::duration<double, std::milli>(end - start).count();
	std::cout << "실행 시간: " << ms << " ms" << std::endl;
	glTexImage2D(GL_TEXTURE_2D, 0, 3, WIDTH, HEIGHT, 0, GL_RGB,
		GL_UNSIGNED_BYTE, &MyTexture[0][0][0]);
	return ms;
}

// 초기화 : 데이터 f, iso(파일값) -> CPU 할당 -> 읽기·이진화 -> 블록 -> GPU 할당·복사 (d_N 은 4normal 판에서 따로)
void MyInit(int f, int iso) {
	CUR = f;
	VOLX = FILE_X[f]; VOLY = FILE_Y[f]; VOLZ = FILE_Z[f];
	ISO = iso;
	BZ_COUNT = VOLZ / BSIZE + (VOLZ % BSIZE != 0);
	BY_COUNT = VOLY / BSIZE + (VOLY % BSIZE != 0);
	BX_COUNT = VOLX / BSIZE + (VOLX % BSIZE != 0);
	size_t nVox = (size_t)VOLX * VOLY * VOLZ;
	size_t nBlk = (size_t)BZ_COUNT * BY_COUNT * BX_COUNT;
	const char* TYPE_NAME[3] = { "unsigned char", "unsigned short", "signed short" };
	printf("\n========== 데이터 %d : %s (%d x %d x %d, %s, iso(파일값) = %d) ==========\n",
		f + 1, FILE_NAME[f], VOLX, VOLY, VOLZ, TYPE_NAME[FILE_TYPE[f]], ISO);
	printf("  블록 %d x %d x %d\n", BX_COUNT, BY_COUNT, BZ_COUNT);

	vol = (unsigned char*)malloc(nVox);
	bM = (unsigned char*)malloc(nBlk);
	FileRead(f);
	printf("binarized (ISO = %d, inside : d >= ISO)\n", ISO);

	GenBlocks();

	COV_BYTES = nVox * 6 * sizeof(float);
	GpuSetDim(VOLX, VOLY, VOLZ, BY_COUNT, BX_COUNT);
	CudaCheck(cudaGetLastError(), "크기 설정");
	CudaCheck(cudaMalloc(&d_vol, nVox), "d_vol 할당");
	CudaCheck(cudaMemcpy(d_vol, vol, nVox, cudaMemcpyHostToDevice), "d_vol 복사");
	CudaCheck(cudaMalloc(&d_bM, nBlk), "d_bM 할당");
	CudaCheck(cudaMemcpy(d_bM, bM, nBlk, cudaMemcpyHostToDevice), "d_bM 복사");
	CudaCheck(cudaMalloc(&d_A, COV_BYTES), "d_A 할당");
	CudaCheck(cudaMemset(d_A, 0, COV_BYTES), "d_A 초기화");
	CudaCheck(cudaMalloc(&d_B, COV_BYTES), "d_B 할당");
	CudaCheck(cudaMemset(d_B, 0, COV_BYTES), "d_B 초기화");
	CudaCheck(cudaMalloc(&d_img, WIDTH * HEIGHT * 3), "d_img 할당");
	std::cout << "GPU 메모리 : A, B 각 " << COV_BYTES / (1024 * 1024) << " MB" << std::endl;
}

void MyFree() {
	CudaCheck(cudaFree(d_vol), "d_vol 해제");
	CudaCheck(cudaFree(d_bM), "d_bM 해제");
	CudaCheck(cudaFree(d_A), "d_A 해제");
	CudaCheck(cudaFree(d_B), "d_B 해제");
	CudaCheck(cudaFree(d_N), "d_N 해제");
	CudaCheck(cudaFree(d_img), "d_img 해제");
	d_vol = 0; d_bM = 0; d_A = 0; d_B = 0; d_N = 0; d_img = 0;
	free(vol); free(bM);
	vol = 0; bM = 0;
	std::cout << "해제 완료" << std::endl;
}

// (실험) 1단계 전체 (기존 main 의 1단계 반복과 같음). 반환 = ms
double RunStep1() {
	auto s = std::chrono::high_resolution_clock::now();
	for (int z = 0; z < VOLZ; z++) {
		GpuStep1(d_vol, d_A, z, R_KER, SIGMA);
		CudaCheck(cudaGetLastError(), "1단계 실행");
	}
	CudaCheck(cudaDeviceSynchronize(), "1단계 수행");
	auto e = std::chrono::high_resolution_clock::now();
	double ms = std::chrono::duration<double, std::milli>(e - s).count();
	printf("1단계 (R=%d) : %.1f ms\n", R_KER, ms);
	return ms;
}

// (실험) 2단계 전체 (기존 main 의 2단계 반복과 같음). d_B 를 0 으로 비운 뒤 시작 (비우기는 시간에서 뺌). 반환 = ms
double RunStep2(float rhoLo, float rhoHi) {
	CudaCheck(cudaMemset(d_B, 0, COV_BYTES), "d_B 초기화");
	CudaCheck(cudaDeviceSynchronize(), "d_B 초기화 수행");
	auto s = std::chrono::high_resolution_clock::now();
	for (int z = 0; z < VOLZ; z++) {
		GpuStep2(d_vol, d_A, d_B, z, SIGMA_N, SIGMA_T, SIGMA_R, rhoLo, rhoHi);
		CudaCheck(cudaGetLastError(), "2단계 실행");
		if (z % 10 == 0) {
			CudaCheck(cudaDeviceSynchronize(), "2단계 수행");
			std::cout << "  2단계 z = " << z << " / " << VOLZ << std::endl;
		}
	}
	CudaCheck(cudaDeviceSynchronize(), "2단계 수행");
	auto e = std::chrono::high_resolution_clock::now();
	double ms = std::chrono::duration<double, std::milli>(e - s).count();
	printf("2단계 (rho %.2f ~ %.2f) : %.1f ms\n", rhoLo, rhoHi, ms);
	return ms;
}

// (실험) 2차 공분산 -> 법선 배열. 반환 = ms
double RunToNormal() {
	auto s = std::chrono::high_resolution_clock::now();
	GpuToNormal(d_B, d_N);
	CudaCheck(cudaGetLastError(), "법선 변환 실행");
	CudaCheck(cudaDeviceSynchronize(), "법선 변환 수행");
	auto e = std::chrono::high_resolution_clock::now();
	double ms = std::chrono::duration<double, std::milli>(e - s).count();
	printf("법선 변환 : %.1f ms\n", ms);
	return ms;
}

// (실험) 띠 통계 : 1차 공분산이 0 이 아닌 격자점 수, 그중 rho 구간 비율 (1단계 직후 호출)
void LogBand(FILE* fBand, int f, int iso) {
	unsigned long long cnt[4];
	GpuStats(d_A, cnt, RHO_LO, RHO_HI);
	double nVox = (double)VOLX * VOLY * VOLZ;
	double band = (double)cnt[0];
	double r0 = (band > 0) ? cnt[1] / band : 0, r1 = (band > 0) ? cnt[2] / band : 0, r2 = (band > 0) ? cnt[3] / band : 0;
	printf("[띠] 데이터 %d iso %d : 띠 안 %llu (%.2f%%), rho < %.2f : %.2f%%, %.2f~%.2f : %.2f%%, >= %.2f : %.2f%%\n",
		f + 1, iso, cnt[0], band / nVox * 100, RHO_LO, r0 * 100, RHO_LO, RHO_HI, r1 * 100, RHO_HI, r2 * 100);
	fprintf(fBand, "%d,%d,%llu,%.6f,%.6f,%.6f,%.6f\n", f + 1, iso, cnt[0], band / nVox, r0, r1, r2);
	fflush(fBand);
}

// 마지막 사진을 창에 띄우기만 함
void MyDisplay() {
	glClear(GL_COLOR_BUFFER_BIT);
	glBegin(GL_QUADS);
	float fSize = 1.0f;
	glTexCoord2f(0.0, 0.0); glVertex3f(-fSize, -fSize, 0.0);
	glTexCoord2f(0.0, 1.0); glVertex3f(-fSize, fSize, 0.0);
	glTexCoord2f(1.0, 1.0); glVertex3f(fSize, fSize, 0.0);
	glTexCoord2f(1.0, 0.0); glVertex3f(fSize, -fSize, 0.0);
	glEnd();
	glutSwapBuffers();
}

int main(int argc, char** argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutCreateWindow("OpenGL Drawing Example");
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
	glEnable(GL_TEXTURE_2D);
	printf("파라미터 : R = %d, sigma = %.2f, sn = %.2f, st = %.2f, sr = %.2f, rho %.2f ~ %.2f\n",
		R_KER, SIGMA, SIGMA_N, SIGMA_T, SIGMA_R, RHO_LO, RHO_HI);

	// (실험) 로그 파일 4개 (머리줄은 영어 : 엑셀 한글 깨짐 방지)
	FILE* fRun = fopen("log_run.csv", "w");
	fprintf(fRun, "data,exp,variant,view,iso,step1_ms,step2_ms,normal_ms,render_ms,render_array_MB\n");
	FILE* fBand = fopen("log_band.csv", "w");
	fprintf(fBand, "data,iso,band_voxels,band_ratio,rho_lt_lo,rho_lo_to_hi,rho_ge_hi\n");
	FILE* fPair = fopen("log_pair.csv", "w");
	fprintf(fPair, "data,view,iso,pair,changed_px,surface_px,changed_pct,silhouette_px\n");
	FILE* fRef = fopen("log_ref.csv", "w");
	fprintf(fRef, "data,view,ref_file,diff_px(-1=no file -2=size)\n");

	size_t nPix3 = (size_t)WIDTH * HEIGHT * 3;
	std::vector<unsigned char> img[VAR_COUNT][2];      // 판 x 시점 사진 (쌍 비교용)
	for (int k = 0; k < VAR_COUNT; k++)
		for (int vw = 0; vw < 2; vw++) img[k][vw].resize(nPix3);
	std::vector<unsigned char> baseFront(nPix3);       // 실험 B 재사용용 (3base, 시점 0)
	char name[256];

	for (int f = 0; f < FILE_COUNT; f++) {
		//==================== 실험 A : 판 비교 (iso 고정) ====================
		int iso = A_ISO[f];
		MyInit(f, iso);
		double t1 = RunStep1();
		LogBand(fBand, f, iso);
		double covMB = COV_BYTES / (1024.0 * 1024.0);
		double t2Base = 0;

		for (int k = 0; k < VAR_COUNT; k++) {
			printf("\n----- 실험 A : 데이터 %d, 판 %s -----\n", f + 1, VAR_NAME[k]);
			double t2 = 0, tN = 0, MB = covMB;
			const float* arr = d_A;
			int useN = 0;
			if (k == 1) { t2 = RunStep2(NOMIX_LO, NOMIX_HI); arr = d_B; }
			if (k == 2) { t2 = RunStep2(RHO_LO, RHO_HI); t2Base = t2; arr = d_B; }
			if (k == 3) {
				// d_B = 바로 앞 3base 의 2차 공분산 그대로. 1차 공분산은 더 안 쓰므로 해제 후 법선 배열 할당
				CudaCheck(cudaFree(d_A), "d_A 해제"); d_A = 0;
				CudaCheck(cudaMalloc(&d_N, COV_BYTES / 2), "d_N 할당");
				CudaCheck(cudaMemset(d_N, 0, COV_BYTES / 2), "d_N 초기화");
				t2 = t2Base;                 // 법선 판도 3base 의 2단계를 거친 것
				tN = RunToNormal();
				arr = d_N; useN = 1; MB = covMB / 2;
			}
			for (int vw = 0; vw < VIEW_COUNT[f]; vw++) {
				double tR = Render(vw, arr, useN);
				memcpy(img[k][vw].data(), &MyTexture[0][0][0], nPix3);
				snprintf(name, sizeof(name), "%d_A_%s_%s_iso%d.bmp", f + 1, VAR_NAME[k], VIEW_NAME[f][vw], iso);
				SaveBMP(name, img[k][vw].data());
				fprintf(fRun, "%d,A,%s,%s,%d,%.1f,%.1f,%.1f,%.1f,%.1f\n",
					f + 1, VAR_NAME[k], VIEW_NAME[f][vw], iso, t1, t2, tN, tR, MB);
				fflush(fRun);

				// base 재현 확인 : 기존 시점(정면) 사진과 비교
				if (k == 2 && vw == 0) {
					char ref[256];
					snprintf(ref, sizeof(ref), "%s_%d.bmp", REF_NAME, f + 1);
					long long dpx = CompareWithFile(ref, img[k][vw].data());
					if (dpx == -1)      printf("[재현] %s 없음 -> 비교 생략\n", ref);
					else if (dpx == -2) printf("[재현] %s 크기 다름\n", ref);
					else                printf("[재현] %s 와 다른 픽셀 %lld 개 %s\n", ref, dpx, (dpx == 0) ? "(일치)" : "(불일치!)");
					fprintf(fRef, "%d,%s,%s,%lld\n", f + 1, VIEW_NAME[f][vw], ref, dpx);
					fflush(fRef);
				}
			}
		}

		// 쌍 비교 (사슬) : 판 k -> 판 k+1
		for (int vw = 0; vw < VIEW_COUNT[f]; vw++)
			for (int p = 0; p < VAR_COUNT - 1; p++) {
				long long changed, surf, sil;
				snprintf(name, sizeof(name), "%d_Adiff_%s_%s_iso%d.bmp", f + 1, PAIR_NAME[p], VIEW_NAME[f][vw], iso);
				ComparePair(img[p][vw].data(), img[p + 1][vw].data(), name, changed, surf, sil);
				double pct = (surf > 0) ? 100.0 * changed / surf : 0;
				printf("[쌍] 데이터 %d %s %s : 바뀐 픽셀 %lld / %lld (%.2f%%), 실루엣 %lld\n",
					f + 1, VIEW_NAME[f][vw], PAIR_NAME[p], changed, surf, pct, sil);
				fprintf(fPair, "%d,%s,%d,%s,%lld,%lld,%.4f,%lld\n", f + 1, VIEW_NAME[f][vw], iso, PAIR_NAME[p], changed, surf, pct, sil);
				fflush(fPair);
			}
		memcpy(baseFront.data(), img[2][0].data(), nPix3);
		MyFree();

		//==================== 실험 B : iso 따로 (base, 정면만) ====================
		if (B_DO[f] == 0) continue;
		for (int i = 0; i < B_ISO_COUNT; i++) {
			int isoB = B_ISO[f][i];
			snprintf(name, sizeof(name), "%d_B_iso%d_3base_front.bmp", f + 1, isoB);
			if (isoB == A_ISO[f]) {          // 실험 A 와 같은 조건 -> 다시 돌리지 않고 재사용
				SaveBMP(name, baseFront.data());
				fprintf(fRun, "%d,B,3base,front,%d,-1,-1,-1,-1,-1\n", f + 1, isoB);   // -1 = A 재사용
				fflush(fRun);
				continue;
			}
			printf("\n----- 실험 B : 데이터 %d, iso %d -----\n", f + 1, isoB);
			MyInit(f, isoB);
			double tb1 = RunStep1();
			LogBand(fBand, f, isoB);
			double tb2 = RunStep2(RHO_LO, RHO_HI);
			double tbR = Render(0, d_B, 0);
			SaveBMP(name, &MyTexture[0][0][0]);
			fprintf(fRun, "%d,B,3base,front,%d,%.1f,%.1f,0.0,%.1f,%.1f\n",
				f + 1, isoB, tb1, tb2, tbR, COV_BYTES / (1024.0 * 1024.0));
			fflush(fRun);
			MyFree();
		}
	}

	fclose(fRun); fclose(fBand); fclose(fPair); fclose(fRef);
	printf("\n========== 전체 완료 : log_run.csv, log_band.csv, log_pair.csv, log_ref.csv ==========\n");

	glutDisplayFunc(MyDisplay);
	glutMainLoop();
	return 0;
}
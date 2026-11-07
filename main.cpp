// ============================================================================
// exp4_main.cpp : 체크리스트 3번 - 단일 임계 vs 섞기 (일회성 실험 코드)
//  판 6개 (모두 2차 1회, 공분산 저장, 2단계의 rho 구간만 다름)
//    none     : 혼합 없음 (rho 2.0 ~ 3.0 -> 섞는 비율 항상 0)          = 실험1 2aniso
//    hard0.25 ~ hard0.40 : 단일 임계 (rho T ~ T+0.0001 -> 임계 이상이면 구형 커널)
//    blend    : 섞기 0.25 ~ 0.40                                       = 실험1 3base
//  쌍 : 혼합 효과(none->hard0.30), 단일 vs 섞기(hard0.30->blend), 임계 흔들림(hard0.30->hard0.25, hard0.30->hard0.35)
//  시점 : 정면만 (1번은 기존 시점). 로그 : t_run.csv, t_pair.csv, t_ref.csv
//  커널 파일 exp_cumian.cu 그대로
// (아래는 exp2_main 머리 주석, 참고용)
//  기준 : exp_main.cpp (실험1). 커널 파일은 exp_cumian.cu 그대로 (수정 없음)
//  판 4개 (쌍 = 저장 방식만 다름)
//    2aniso        : 혼합 없는 2차 공분산 -> 공분산 렌더
//    2aniso_normal : 위와 같은 2차 공분산 -> 법선 변환 -> 법선 렌더   (새로 추가)
//    3base         : 혼합 2차 공분산 -> 공분산 렌더
//    4normal       : 위와 같은 2차 공분산 -> 법선 변환 -> 법선 렌더
//  쌍 : pS_nomix (2aniso -> 2aniso_normal), pS_mix (3base -> 4normal)
//  확인 : 네 판 모두 실험1 사진(%d_A_%s_%s_iso%d.bmp)이 폴더에 있으면 픽셀 비교 (2aniso_normal 은 실험1에 없음)
//  바뀐 것 (exp_main 대비)
//    - 판 목록, 쌍 목록, 실험 B 제거
//    - 법선 변환 전에 d_N 을 0 으로 비움 (K_ToNormal 은 0 이 아닌 격자점만 쓰므로, 비우지 않으면 앞 판 법선이 남음)
//    - d_A 를 해제하지 않음 (3base 가 2aniso_normal 뒤에 d_A 를 다시 씀)
//  로그 : s_run.csv, s_pair.csv, s_ref.csv
// ============================================================================
#define _CRT_SECURE_NO_WARNINGS
#include <GL/glut.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>

#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

#define WINDOW_WIDTH 1024
#define WINDOW_HEIGHT 1024
#define WIDTH   1024
#define HEIGHT  1024
const int BSIZE = 8;

//==================== 데이터별 전역 파라미터 (exp_main 과 같음) ====================
const int   FILE_COUNT = 3;
const char* FILE_NAME[FILE_COUNT] = { "bighead.den", "abdomen300.den", "output_HEAD_01" };
const int   FILE_TYPE[FILE_COUNT] = { 0, 1, 2 };
const int   FILE_X[FILE_COUNT] = { 256, 512, 512 };
const int   FILE_Y[FILE_COUNT] = { 256, 512, 512 };
const int   FILE_Z[FILE_COUNT] = { 225, 300, 552 };
const int   A_ISO[FILE_COUNT] = { 120, 1274, 250 };

const glm::vec3 FILE_EYE_OFS[FILE_COUNT] = { glm::vec3(-128, -128, -12), glm::vec3(0, -400, 0), glm::vec3(0, +400, 0) };
const glm::vec3 FILE_UP[FILE_COUNT] = { glm::vec3(0, 1, 0),         glm::vec3(0, 0, 1),   glm::vec3(0, 0, -1) };
const float     FILE_PIXEL[FILE_COUNT] = { 0.5f, 0.5f, 0.6f };
const int   VIEW_COUNT[FILE_COUNT] = { 1, 2, 2 };
const char* VIEW_NAME[FILE_COUNT][2] = { { "orig", "" }, { "front", "rot45" }, { "front", "rot45" } };
const float VIEW_DEG[2] = { 0.0f, 45.0f };

// (실험4) 판 : 2단계 rho 구간 [LO, HI]. 섞는 비율 = smoothstep(LO, HI, rho)
const int   VAR_COUNT = 6;
const char* VAR_NAME[VAR_COUNT] = { "none", "hard0.25", "hard0.30", "hard0.35", "hard0.40", "blend" };
const float VAR_LO[VAR_COUNT] = { 2.0f, 0.25f,   0.30f,   0.35f,   0.40f,   0.25f };
const float VAR_HI[VAR_COUNT] = { 3.0f, 0.2501f, 0.3001f, 0.3501f, 0.4001f, 0.40f };
const char* VAR_REF[VAR_COUNT] = { "2aniso", "", "", "", "", "3base" };   // 실험1 같은 판 (재현 확인용)
const int   PAIR_COUNT = 4;
const char* PAIR_NAME[PAIR_COUNT] = { "pT_mixon", "pT_hard_blend", "pT_030_025", "pT_030_035" };
const int   PAIR_A[PAIR_COUNT] = { 0, 2, 2, 2 };
const int   PAIR_B[PAIR_COUNT] = { 2, 5, 1, 3 };

int CUR = 0;
int VOLX, VOLY, VOLZ;
int BZ_COUNT, BY_COUNT, BX_COUNT;
int ISO;
size_t COV_BYTES = 0;

unsigned char MyTexture[HEIGHT][WIDTH][3];
unsigned char* vol = 0;
unsigned char* bM = 0;

using namespace std;

const int   R_KER = 3;
const float SIGMA = R_KER / 3.0f;
const float SIGMA_N = 0.6f;
const float SIGMA_T = 4.0f;
const float SIGMA_R = 3.0f;
const float RHO_LO = 0.25f;
const float RHO_HI = 0.40f;
const float NOMIX_LO = 2.0f;     // 혼합 없음 : rho 는 최대 1 이므로 섞는 비율 항상 0
const float NOMIX_HI = 3.0f;

unsigned char* d_vol = 0;
unsigned char* d_bM = 0;
float* d_A = 0;
float* d_B = 0;
float* d_N = 0;
unsigned char* d_img = 0;

void GpuSetDim(int volx, int voly, int volz, int byCount, int bxCount);
void GpuStep1(const unsigned char* d_vol, float* d_A, int z, int R, float sigma);
void GpuStep2(const unsigned char* d_vol, const float* d_A, float* d_B, int z,
	float sn, float st, float sr, float rhoLo, float rhoHi);
void GpuRender(const unsigned char* d_vol, const unsigned char* d_bM, const float* d_B, unsigned char* d_img,
	const float eye[3], const float u[3], const float v[3], const float w[3], float pixel);
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
}

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

// 기존 사진과 픽셀 비교 : -1 = 파일 없음, -2 = 크기 다름, 0 이상 = 다른 픽셀 수
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

// 쌍 비교 (exp_main 과 같은 정의) : 바뀐 픽셀 = 밝기 차 > 8, 분모 = 둘 중 하나라도 배경 아님
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

glm::vec3 RotateAbout(glm::vec3 v, glm::vec3 k, float deg) {
	float a = deg * 3.14159265f / 180.0f;
	return v * cosf(a) + glm::cross(k, v) * sinf(a) + k * glm::dot(k, v) * (1.0f - cosf(a));
}

// 렌더 (exp_main 과 같음) : useN = 0 공분산 배열, 1 법선 배열
double Render(int view, const float* arr, int useN) {
	glm::vec3 at(VOLX / 2, VOLY / 2, VOLZ / 2);
	glm::vec3 ofs = FILE_EYE_OFS[CUR];
	if (view != 0) ofs = RotateAbout(ofs, glm::normalize(FILE_UP[CUR]), VIEW_DEG[view]);
	glm::vec3 eye = at + ofs;
	glm::vec3 up = FILE_UP[CUR];

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
	std::cout << "렌더 : " << ms << " ms" << std::endl;
	glTexImage2D(GL_TEXTURE_2D, 0, 3, WIDTH, HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, &MyTexture[0][0][0]);
	return ms;
}

// 초기화 : d_N 도 여기서 함께 할당 (법선 3성분 = 공분산의 절반)
void MyInit(int f, int iso) {
	CUR = f;
	VOLX = FILE_X[f]; VOLY = FILE_Y[f]; VOLZ = FILE_Z[f];
	ISO = iso;
	BZ_COUNT = VOLZ / BSIZE + (VOLZ % BSIZE != 0);
	BY_COUNT = VOLY / BSIZE + (VOLY % BSIZE != 0);
	BX_COUNT = VOLX / BSIZE + (VOLX % BSIZE != 0);
	size_t nVox = (size_t)VOLX * VOLY * VOLZ;
	size_t nBlk = (size_t)BZ_COUNT * BY_COUNT * BX_COUNT;
	printf("\n========== 데이터 %d : %s (%d x %d x %d, iso(파일값) = %d) ==========\n",
		f + 1, FILE_NAME[f], VOLX, VOLY, VOLZ, ISO);

	vol = (unsigned char*)malloc(nVox);
	bM = (unsigned char*)malloc(nBlk);
	FileRead(f);
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
	CudaCheck(cudaMalloc(&d_N, COV_BYTES / 2), "d_N 할당");
	CudaCheck(cudaMalloc(&d_img, WIDTH * HEIGHT * 3), "d_img 할당");
	std::cout << "GPU 메모리 : A, B 각 " << COV_BYTES / (1024 * 1024) << " MB, N " << COV_BYTES / 2 / (1024 * 1024) << " MB" << std::endl;
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
}

double RunStep1() {
	auto s = std::chrono::high_resolution_clock::now();
	for (int z = 0; z < VOLZ; z++) {
		GpuStep1(d_vol, d_A, z, R_KER, SIGMA);
		CudaCheck(cudaGetLastError(), "1단계 실행");
	}
	CudaCheck(cudaDeviceSynchronize(), "1단계 수행");
	auto e = std::chrono::high_resolution_clock::now();
	double ms = std::chrono::duration<double, std::milli>(e - s).count();
	printf("1단계 : %.1f ms\n", ms);
	return ms;
}

// 2단계 : d_B 를 0 으로 비운 뒤 계산 (비우기는 시간에서 뺌)
double RunStep2(float rhoLo, float rhoHi) {
	CudaCheck(cudaMemset(d_B, 0, COV_BYTES), "d_B 초기화");
	CudaCheck(cudaDeviceSynchronize(), "d_B 초기화 수행");
	auto s = std::chrono::high_resolution_clock::now();
	for (int z = 0; z < VOLZ; z++) {
		GpuStep2(d_vol, d_A, d_B, z, SIGMA_N, SIGMA_T, SIGMA_R, rhoLo, rhoHi);
		CudaCheck(cudaGetLastError(), "2단계 실행");
	}
	CudaCheck(cudaDeviceSynchronize(), "2단계 수행");
	auto e = std::chrono::high_resolution_clock::now();
	double ms = std::chrono::duration<double, std::milli>(e - s).count();
	printf("2단계 (rho %.2f ~ %.2f) : %.1f ms\n", rhoLo, rhoHi, ms);
	return ms;
}

// (실험2) 법선 변환 : d_N 을 0 으로 비운 뒤 d_B -> d_N (비우기는 시간에서 뺌)
double RunToNormal() {
	CudaCheck(cudaMemset(d_N, 0, COV_BYTES / 2), "d_N 초기화");
	CudaCheck(cudaDeviceSynchronize(), "d_N 초기화 수행");
	auto s = std::chrono::high_resolution_clock::now();
	GpuToNormal(d_B, d_N);
	CudaCheck(cudaGetLastError(), "법선 변환 실행");
	CudaCheck(cudaDeviceSynchronize(), "법선 변환 수행");
	auto e = std::chrono::high_resolution_clock::now();
	double ms = std::chrono::duration<double, std::milli>(e - s).count();
	printf("법선 변환 : %.1f ms\n", ms);
	return ms;
}

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

	FILE* fRun = fopen("t_run.csv", "w");
	fprintf(fRun, "data,variant,view,iso,rho_lo,rho_hi,step1_ms,step2_ms,render_ms\n");
	FILE* fPair = fopen("t_pair.csv", "w");
	fprintf(fPair, "data,view,iso,pair,changed_px,surface_px,changed_pct,silhouette_px\n");
	FILE* fRef = fopen("t_ref.csv", "w");
	fprintf(fRef, "data,variant,ref_file,diff_px(-1=no file -2=size)\n");

	size_t nPix3 = (size_t)WIDTH * HEIGHT * 3;
	std::vector<unsigned char> img[VAR_COUNT];
	for (int k = 0; k < VAR_COUNT; k++) img[k].resize(nPix3);
	char name[256];

	for (int f = 0; f < FILE_COUNT; f++) {
		int iso = A_ISO[f];
		int vw = 0;                                   // 정면 (1번은 기존 시점)
		MyInit(f, iso);
		double t1 = RunStep1();

		for (int k = 0; k < VAR_COUNT; k++) {
			printf("\n----- 데이터 %d, 판 %s -----\n", f + 1, VAR_NAME[k]);
			double t2 = RunStep2(VAR_LO[k], VAR_HI[k]);
			double tR = Render(vw, d_B, 0);
			memcpy(img[k].data(), &MyTexture[0][0][0], nPix3);
			snprintf(name, sizeof(name), "%d_T_%s_%s_iso%d.bmp", f + 1, VAR_NAME[k], VIEW_NAME[f][vw], iso);
			SaveBMP(name, img[k].data());
			fprintf(fRun, "%d,%s,%s,%d,%.4f,%.4f,%.1f,%.1f,%.1f\n", f + 1, VAR_NAME[k], VIEW_NAME[f][vw], iso, VAR_LO[k], VAR_HI[k], t1, t2, tR);
			fflush(fRun);

			if (VAR_REF[k][0] != 0) {                 // 실험1 사진과 비교 (같은 판이면 0)
				char ref[256];
				snprintf(ref, sizeof(ref), "%d_A_%s_%s_iso%d.bmp", f + 1, VAR_REF[k], VIEW_NAME[f][vw], iso);
				long long dpx = CompareWithFile(ref, img[k].data());
				printf("[재현] %s : %lld\n", ref, dpx);
				fprintf(fRef, "%d,%s,%s,%lld\n", f + 1, VAR_NAME[k], ref, dpx);
				fflush(fRef);
			}
		}

		for (int p = 0; p < PAIR_COUNT; p++) {
			long long changed, surf, sil;
			snprintf(name, sizeof(name), "%d_Tdiff_%s_%s_iso%d.bmp", f + 1, PAIR_NAME[p], VIEW_NAME[f][vw], iso);
			ComparePair(img[PAIR_A[p]].data(), img[PAIR_B[p]].data(), name, changed, surf, sil);
			double pct = (surf > 0) ? 100.0 * changed / surf : 0;
			printf("[쌍] 데이터 %d %s : 바뀐 픽셀 %lld / %lld (%.2f%%)\n", f + 1, PAIR_NAME[p], changed, surf, pct);
			fprintf(fPair, "%d,%s,%d,%s,%lld,%lld,%.4f,%lld\n", f + 1, VIEW_NAME[f][vw], iso, PAIR_NAME[p], changed, surf, pct, sil);
			fflush(fPair);
		}
		MyFree();
	}
	 
	fclose(fRun); fclose(fPair); fclose(fRef);
	printf("\n========== 완료 : t_run.csv, t_pair.csv, t_ref.csv ==========\n");

	glutDisplayFunc(MyDisplay);
	glutMainLoop();
	return 0;
}
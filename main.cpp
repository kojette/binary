// ============================================================================
// exp5_main.cpp : 체크리스트 4번 - 비교 방법(중앙차분, 가우시안 평활) 실험 (일회성 실험 코드)
//  커널 파일 : exp5_cumian.cu (exp_cumian + 비교 방법 커널 3개)
//  방법 6개. 표면(이진 바이섹션 교점)·광선·조명·시점은 모두 같고 "법선만" 다르다
//    proposed : 제안 (2차 1회, 섞기 0.25~0.40, 공분산 저장) = 실험1 3base
//    cd       : 이진 볼륨의 중앙차분 (실시간, 추가 메모리 0)
//    gaussR   : 이진 볼륨을 가우시안 평활(반경 R, sigma = R/3, 프로젝트의 3시그마 반경 규칙)한 실수 볼륨의 중앙차분
//               R = 2, 3, 4, 5. 평활 볼륨은 미리 계산해 저장 (복셀당 실수 1개)
//  시점 : 실험1과 같음 (1번 기존 시점, 2·3번 정면 + 45도) + 실험3의 8배 확대 3곳
//  쌍 : 각 비교 방법 -> proposed (정면 / 기존 시점만, 차이 영상 + 바뀐 픽셀)
//  확인 : proposed 는 실험1 3base 사진과 0 픽셀이어야 함 (폴더에 있을 때)
//  로그 : c_run.csv (방법별 전처리 ms, 렌더 ms, 저장 MB), c_pair.csv, c_ref.csv
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

// (실험3) 확대 영역 : 데이터 번호(0부터), 정면 사진 픽셀 좌표 (cx, cy, 그림 뷰어 기준), 이름
const int   ZOOM = 8;                          // 8배 : 두경부 픽셀 간격 0.6 -> 0.075 복셀 (복셀 하나 = 약 13 픽셀)
const int   ZONE_COUNT = 3;
const int   ZONE_DATA[ZONE_COUNT] = { 2, 2, 1 };
const int   ZONE_CX[ZONE_COUNT] = { 625, 510, 340 };
const int   ZONE_CY[ZONE_COUNT] = { 305, 710, 520 };
const char* ZONE_NAME[ZONE_COUNT] = { "eye", "teeth", "spine" };

// (실험5) 방법 : 0 = 제안, 1 = 중앙차분, 2~5 = 가우시안 반경 2~5
const int   METHOD_COUNT = 6;
const char* METHOD_NAME[METHOD_COUNT] = { "proposed", "cd", "gauss2", "gauss3", "gauss4", "gauss5" };
const int   METHOD_R[METHOD_COUNT] = { 0, 0, 2, 3, 4, 5 };

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
float* d_g1 = 0;            // (실험5) 실수 볼륨 (평활 작업용 두 장)
float* d_g2 = 0;
unsigned char* d_img = 0;

void GpuSetDim(int volx, int voly, int volz, int byCount, int bxCount);
void GpuStep1(const unsigned char* d_vol, float* d_A, int z, int R, float sigma);
void GpuStep2(const unsigned char* d_vol, const float* d_A, float* d_B, int z,
	float sn, float st, float sr, float rhoLo, float rhoHi);
void GpuRender(const unsigned char* d_vol, const unsigned char* d_bM, const float* d_B, unsigned char* d_img,
	const float eye[3], const float u[3], const float v[3], const float w[3], float pixel);
void GpuToNormal(const float* d_B, float* d_N);
void GpuToFloat(const unsigned char* d_vol, float* d_g);
void GpuBlur1D(const float* d_in, float* d_out, int axis, int R, float sigma);
void GpuRenderG(const unsigned char* d_vol, const unsigned char* d_bM, const float* d_g, unsigned char* d_img, float* d_out,
	const float eye[3], const float u[3], const float v[3], const float w[3], float pixel);
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
	else           GpuRenderG(d_vol, d_bM, arr, d_img, 0, e3, u3, v3, w3, FILE_PIXEL[CUR]);   // (실험5) 1 = 실수 볼륨 그래디언트
	CudaCheck(cudaGetLastError(), "렌더 실행");
	CudaCheck(cudaDeviceSynchronize(), "렌더 수행");
	CudaCheck(cudaMemcpy(&MyTexture[0][0][0], d_img, nPix * 3, cudaMemcpyDeviceToHost), "이미지 복사");
	auto end = std::chrono::high_resolution_clock::now();
	double ms = std::chrono::duration<double, std::milli>(end - start).count();
	std::cout << "렌더 : " << ms << " ms" << std::endl;
	glTexImage2D(GL_TEXTURE_2D, 0, 3, WIDTH, HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, &MyTexture[0][0][0]);
	return ms;
}

// (실험3) 확대 렌더 : 정면과 같은 방향. 눈을 화면 평면 위에서 관심 영역 중심으로 옮기고, 픽셀 간격을 1/ZOOM 로
//  광선 시작점 = eye + u*(x-512)*pixel + v*(y-512)*pixel (커널과 같음)
//  사진 저장은 y = 0 줄을 파일 맨 앞(= 그림 뷰어의 맨 아래)에 쓰므로, 뷰어 좌표 cy -> 렌더 y = 1023 - cy
double RenderZoom(int cx, int cy, const float* arr, int useN) {
	glm::vec3 at(VOLX / 2, VOLY / 2, VOLZ / 2);
	glm::vec3 eye = at + FILE_EYE_OFS[CUR];
	glm::vec3 up = FILE_UP[CUR];
	glm::vec3 w = glm::normalize(at - eye);
	glm::vec3 u = glm::normalize(glm::cross(up, w));
	glm::vec3 v = glm::normalize(glm::cross(w, u));
	float pixel0 = FILE_PIXEL[CUR];
	int ry = (HEIGHT - 1) - cy;
	eye = eye + u * ((cx - WIDTH * 0.5f) * pixel0) + v * ((ry - HEIGHT * 0.5f) * pixel0);
	float pixel = pixel0 / ZOOM;

	auto start = std::chrono::high_resolution_clock::now();
	float e3[3] = { eye.x, eye.y, eye.z }, u3[3] = { u.x, u.y, u.z }, v3[3] = { v.x, v.y, v.z }, w3[3] = { w.x, w.y, w.z };
	if (useN == 0) GpuRender(d_vol, d_bM, arr, d_img, e3, u3, v3, w3, pixel);
	else           GpuRenderG(d_vol, d_bM, arr, d_img, 0, e3, u3, v3, w3, pixel);       // (실험5) 1 = 실수 볼륨 그래디언트
	CudaCheck(cudaGetLastError(), "확대 렌더 실행");
	CudaCheck(cudaDeviceSynchronize(), "확대 렌더 수행");
	CudaCheck(cudaMemcpy(&MyTexture[0][0][0], d_img, WIDTH * HEIGHT * 3, cudaMemcpyDeviceToHost), "이미지 복사");
	auto end = std::chrono::high_resolution_clock::now();
	double ms = std::chrono::duration<double, std::milli>(end - start).count();
	std::cout << "확대 렌더 : " << ms << " ms" << std::endl;
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
	CudaCheck(cudaFree(d_g1), "d_g1 해제");
	CudaCheck(cudaFree(d_g2), "d_g2 해제");
	d_vol = 0; d_bM = 0; d_A = 0; d_B = 0; d_N = 0; d_img = 0; d_g1 = 0; d_g2 = 0;
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

// (실험5) 실수 볼륨 준비 : R = 0 이면 이진 볼륨 그대로(중앙차분용) -> d_g1
//  R > 0 이면 x, y, z 순서로 한 축씩 평활 : g1 -> g2 -> g1 -> g2. 결과 배열을 돌려줌. 반환 시간 = ms
float* PrepareG(int R, double& ms) {
	auto s = std::chrono::high_resolution_clock::now();
	GpuToFloat(d_vol, d_g1);
	CudaCheck(cudaGetLastError(), "실수 변환 실행");
	float* res = d_g1;
	if (R > 0) {
		float sigma = R / 3.0f;
		GpuBlur1D(d_g1, d_g2, 0, R, sigma);
		CudaCheck(cudaGetLastError(), "평활 x 실행");
		GpuBlur1D(d_g2, d_g1, 1, R, sigma);
		CudaCheck(cudaGetLastError(), "평활 y 실행");
		GpuBlur1D(d_g1, d_g2, 2, R, sigma);
		CudaCheck(cudaGetLastError(), "평활 z 실행");
		res = d_g2;
	}
	CudaCheck(cudaDeviceSynchronize(), "실수 볼륨 준비 수행");
	auto e = std::chrono::high_resolution_clock::now();
	ms = std::chrono::duration<double, std::milli>(e - s).count();
	printf("실수 볼륨 준비 (R = %d) : %.1f ms\n", R, ms);
	return res;
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

	FILE* fRun = fopen("c_run.csv", "w");
	fprintf(fRun, "data,method,view,iso,pre_ms(step1+step2 or blur),render_ms,stored_MB\n");
	FILE* fPair = fopen("c_pair.csv", "w");
	fprintf(fPair, "data,view,iso,method_vs_proposed,changed_px,surface_px,changed_pct\n");
	FILE* fRef = fopen("c_ref.csv", "w");
	fprintf(fRef, "data,view,ref_file,diff_px(-1=no file -2=size)\n");

	size_t nPix3 = (size_t)WIDTH * HEIGHT * 3;
	std::vector<unsigned char> imgP(nPix3), imgM(nPix3);   // 제안(기존 시점/정면) 사진, 비교 방법 사진
	char name[256];

	for (int f = 0; f < FILE_COUNT; f++) {
		int iso = A_ISO[f];
		MyInit(f, iso);
		size_t nVox = (size_t)VOLX * VOLY * VOLZ;

		for (int m = 0; m < METHOD_COUNT; m++) {
			printf("\n----- 데이터 %d, 방법 %s -----\n", f + 1, METHOD_NAME[m]);
			double pre = 0, MB = 0;
			const float* arr = 0;
			int mode = 0;
			if (m == 0) {                                       // 제안 : 1단계 + 2단계(섞기)
				pre = RunStep1() + RunStep2(RHO_LO, RHO_HI);
				arr = d_B; mode = 0; MB = COV_BYTES / (1024.0 * 1024.0);
			}
			else {
				if (m == 1) {                                   // 제안이 끝났으니 공분산 배열 해제 후 실수 볼륨 두 장 할당
					CudaCheck(cudaFree(d_A), "d_A 해제"); d_A = 0;
					CudaCheck(cudaFree(d_B), "d_B 해제"); d_B = 0;
					CudaCheck(cudaMalloc(&d_g1, nVox * sizeof(float)), "d_g1 할당");
					CudaCheck(cudaMalloc(&d_g2, nVox * sizeof(float)), "d_g2 할당");
				}
				arr = PrepareG(METHOD_R[m], pre);
				mode = 1;
				MB = (m == 1) ? 0.0 : nVox * sizeof(float) / (1024.0 * 1024.0);   // 중앙차분은 추가 저장 없음
			}

			// 전체 시점
			for (int vw = 0; vw < VIEW_COUNT[f]; vw++) {
				double tR = Render(vw, arr, mode);
				snprintf(name, sizeof(name), "%d_C_%s_%s_iso%d.bmp", f + 1, METHOD_NAME[m], VIEW_NAME[f][vw], iso);
				SaveBMP(name, &MyTexture[0][0][0]);
				fprintf(fRun, "%d,%s,%s,%d,%.1f,%.1f,%.1f\n", f + 1, METHOD_NAME[m], VIEW_NAME[f][vw], iso, pre, tR, MB);
				fflush(fRun);

				if (vw == 0) {
					if (m == 0) {
						memcpy(imgP.data(), &MyTexture[0][0][0], nPix3);
						char ref[256];
						snprintf(ref, sizeof(ref), "%d_A_3base_%s_iso%d.bmp", f + 1, VIEW_NAME[f][vw], iso);
						long long dpx = CompareWithFile(ref, imgP.data());
						printf("[재현] %s : %lld\n", ref, dpx);
						fprintf(fRef, "%d,%s,%s,%lld\n", f + 1, VIEW_NAME[f][vw], ref, dpx);
						fflush(fRef);
					}
					else {
						memcpy(imgM.data(), &MyTexture[0][0][0], nPix3);
						long long changed, surf, sil;
						snprintf(name, sizeof(name), "%d_Cdiff_%s_vs_proposed_%s_iso%d.bmp", f + 1, METHOD_NAME[m], VIEW_NAME[f][vw], iso);
						ComparePair(imgM.data(), imgP.data(), name, changed, surf, sil);
						double pct = (surf > 0) ? 100.0 * changed / surf : 0;
						printf("[쌍] 데이터 %d %s vs proposed : 바뀐 픽셀 %.2f%%\n", f + 1, METHOD_NAME[m], pct);
						fprintf(fPair, "%d,%s,%d,%s,%lld,%lld,%.4f\n", f + 1, VIEW_NAME[f][vw], iso, METHOD_NAME[m], changed, surf, pct);
						fflush(fPair);
					}
				}
			}
			// 8배 확대 (실험3과 같은 영역)
			for (int z = 0; z < ZONE_COUNT; z++) {
				if (ZONE_DATA[z] != f) continue;
				RenderZoom(ZONE_CX[z], ZONE_CY[z], arr, mode);
				snprintf(name, sizeof(name), "%d_C_%s_%s_x%d_iso%d.bmp", f + 1, METHOD_NAME[m], ZONE_NAME[z], ZOOM, iso);
				SaveBMP(name, &MyTexture[0][0][0]);
			}
		}
		MyFree();
	}

	fclose(fRun); fclose(fPair); fclose(fRef);
	printf("\n========== 완료 : c_run.csv, c_pair.csv, c_ref.csv ==========\n");

	glutDisplayFunc(MyDisplay);
	glutMainLoop();
	return 0;
}
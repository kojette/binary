// main_v1 : 데이터 3개를 차례로 읽어 이진화 -> 전처리(장 단위) -> 렌더 -> 저장
// v1 (메모리) : 1·2단계를 z 한 장씩 돌리고, 2차 공분산은 띠 안 격자점만 저장 (띠 번호 표 + 띠 배열)
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

#define WINDOW_WIDTH 1024
#define WINDOW_HEIGHT 1024
#define WIDTH   1024        // (2차_4차) 512 -> 1024 : 512 데이터 폭 전체가 화면에 들어오도록
#define HEIGHT  1024
const int BSIZE = 8;

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
//  뼈 문턱 250 HU (250 / 300 / 350 비교 후 고정. 2·3 은 히스토그램에 뼈 봉우리가 없어 골짜기를 못 찾음)
const int   FILE_ISO_HU[FILE_COUNT] = { 120, 250, 250 };
const int   FILE_HU_OFS[FILE_COUNT] = { 0, 1024, 0 };
const char* OUT_NAME = "v1";    // 저장 이름 : 데이터번호_v1_slice_base.bmp
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


unsigned char MyTexture[HEIGHT][WIDTH][3];
unsigned char* vol = 0;   // [VOLZ][VOLY][VOLX] 를 한 줄로 : vol[(z * VOLY + y) * VOLX + x]


unsigned char* bM = 0;    // [BZ_COUNT][BY_COUNT][BX_COUNT] 를 한 줄로

using namespace std;


// 1단계 : 등방 PCA
const int   R_KER = 3;                  // 정육면체 창 반경
const float SIGMA = R_KER / 3.0f;       // 가우시안 폭 (3 sigma = R)
// 2단계 : 타원체 PCA (1회)
const float SIGMA_N = 0.6f;             // 법선 방향 폭
const float SIGMA_T = 4.0f;             // 표면 방향 폭
// (base) 1차 법선을 못 믿는 곳은 구 커널 쪽으로 섞기 : 섞는 비율 = smoothstep(RHO_LO, RHO_HI, 1차 λ3/λ2)
const float SIGMA_R = 3.0f;             // 구 커널 폭 (3시그마 반경 9)
const float RHO_LO = 0.25f;             // 1차 λ3/λ2 가 이 이하면 눕힌 타원체 그대로
const float RHO_HI = 0.40f;             // 이 이상이면 구 커널

// GPU 메모리
unsigned char* d_vol = 0;   // 이진 볼륨
unsigned char* d_bM = 0;    // 블록 최대값
unsigned char* d_img = 0;   // 렌더 결과 RGB
// (v1) 장 버퍼 : z 한 장(VOLX * VOLY 격자점)짜리. 장마다 0 으로 비우고 다시 씀
float* d_Aslice = 0;        // 1차 공분산 6성분 (xx, yy, zz, xy, xz, yz), 한 장, 띠 밖 = 0
float* d_Bslice = 0;        // 2차 공분산 6성분, 한 장, 띠 밖 = 0
float* Bslice = 0;          // d_Bslice 를 CPU 로 꺼내 띠를 고르는 자리 (한 장)
// (v1) 띠만 저장 : 렌더가 쓰는 최종 결과
int* d_idx = 0;             // 격자점마다 띠 번호 (띠 밖 = -1), 볼륨 전체
float* d_P = 0;             // 띠 안 격자점의 2차 공분산 6성분만 차곡차곡 : d_P[띠 번호 * 6 + k]
int M_BAND = 0;             // 띠 안 격자점 개수

// cumian_v1.cu 의 창구 함수
void GpuSetDim(int volx, int voly, int volz, int byCount, int bxCount);   // (2차_4차) 볼륨 크기를 GPU 상수로
void GpuStep1(const unsigned char* d_vol, float* d_Aslice, int z, int R, float sigma);        // 1단계 (z 한 장)
void GpuStep2(const unsigned char* d_vol, const float* d_Aslice, float* d_Bslice, int z,
	float sn, float st, float sr, float rhoLo, float rhoHi);                                      // 2단계 (z 한 장)
void GpuRender(const unsigned char* d_vol, const unsigned char* d_bM, const int* d_idx, const float* d_P, unsigned char* d_img,
	const float eye[3], const float u[3], const float v[3], const float w[3], float pixel);      // 렌더 (띠 배열에서 읽음)

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

// (v1) 띠 배열 d_idx + d_P 로 렌더
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
	GpuRender(d_vol, d_bM, d_idx, d_P, d_img, e3, u3, v3, w3, FILE_PIXEL[CUR]);
	CudaCheck(cudaGetLastError(), "렌더 실행");
	CudaCheck(cudaDeviceSynchronize(), "렌더 수행");
	CudaCheck(cudaMemcpy(&MyTexture[0][0][0], d_img, nPix * 3, cudaMemcpyDeviceToHost), "이미지 복사");
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "실행 시간: " << duration.count() * 0.001f << " ms" << std::endl;
	glTexImage2D(GL_TEXTURE_2D, 0, 3, WIDTH, HEIGHT, 0, GL_RGB,
		GL_UNSIGNED_BYTE, &MyTexture[0][0][0]);
}

// (v1) 전처리 : z 한 장씩 1단계 -> 2단계 -> 그 장의 띠만 골라 이어 붙이기
//  장 z 에서 2단계가 1차 공분산을 읽는 곳은 "자기 격자점" 하나뿐 (이웃은 이진 볼륨에서 읽음)
//  -> 1차 공분산은 그 장 것만 있으면 된다 = 장 버퍼 하나로 충분
//  띠 번호는 z 순서, 장 안에서는 (y, x) 순서로 0, 1, 2, ... (= 볼륨 전체 격자점 번호 순서)
void Preprocess() {
	int nSlice = VOLX * VOLY;                  // 한 장의 격자점 수
	size_t nVox = (size_t)nSlice * VOLZ;       // 볼륨 전체 격자점 수
	size_t sliceBytes = (size_t)nSlice * 6 * sizeof(float);

	int* idx = (int*)malloc(nVox * sizeof(int));   // 띠 번호 표 (CPU 에서 채운 뒤 GPU 로)
	std::vector<float> band;                       // 띠 배열 (CPU 에서 이어 붙인 뒤 GPU 로)
	M_BAND = 0;

	auto preStart = std::chrono::high_resolution_clock::now();
	for (int z = 0; z < VOLZ; z++) {
		// 장 버퍼 비우기 : 기존의 "볼륨 전체를 0 으로 초기화" 와 같은 역할 (띠 밖 = 0)
		CudaCheck(cudaMemset(d_Aslice, 0, sliceBytes), "A 장 초기화");
		CudaCheck(cudaMemset(d_Bslice, 0, sliceBytes), "B 장 초기화");

		// 1단계 -> 2단계 (같은 장)
		GpuStep1(d_vol, d_Aslice, z, R_KER, SIGMA);
		CudaCheck(cudaGetLastError(), "1단계 실행");
		GpuStep2(d_vol, d_Aslice, d_Bslice, z, SIGMA_N, SIGMA_T, SIGMA_R, RHO_LO, RHO_HI);
		CudaCheck(cudaGetLastError(), "2단계 실행");

		// 그 장의 2차 공분산을 CPU 로 (cudaMemcpy 는 앞의 커널이 끝날 때까지 기다림)
		CudaCheck(cudaMemcpy(Bslice, d_Bslice, sliceBytes, cudaMemcpyDeviceToHost), "B 장 복사");

		// 띠 고르기 : 6성분 중 하나라도 0 이 아니면 띠 안 (렌더가 "전부 0 이면 건너뜀" 으로 보는 조건과 같음)
		for (int k = 0; k < nSlice; k++) {
			const float* c = Bslice + (size_t)k * 6;
			int nonzero = 0;
			for (int m = 0; m < 6; m++)
				if (c[m] != 0.0f) nonzero = 1;

			size_t i = (size_t)z * nSlice + k;     // 볼륨 전체 격자점 번호
			if (nonzero == 1) {
				idx[i] = M_BAND;
				M_BAND++;
				for (int m = 0; m < 6; m++) band.push_back(c[m]);   // 6성분을 그대로 복사
			}
			else idx[i] = -1;
		}

		if (z % 50 == 0) std::cout << "  전처리 z = " << z << " / " << VOLZ << std::endl;   // [로그]
	}
	auto preEnd = std::chrono::high_resolution_clock::now();
	std::cout << "전처리 (1단계 + 2단계 + 띠 고르기, 장 단위) : "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(preEnd - preStart).count() << " ms" << std::endl;   // [로그]

	// 띠 번호 표와 띠 배열을 GPU 로
	CudaCheck(cudaMalloc(&d_idx, nVox * sizeof(int)), "d_idx 할당");
	CudaCheck(cudaMemcpy(d_idx, idx, nVox * sizeof(int), cudaMemcpyHostToDevice), "d_idx 복사");
	CudaCheck(cudaMalloc(&d_P, (size_t)M_BAND * 6 * sizeof(float)), "d_P 할당");
	CudaCheck(cudaMemcpy(d_P, band.data(), (size_t)M_BAND * 6 * sizeof(float), cudaMemcpyHostToDevice), "d_P 복사");
	free(idx);

	// [로그] 공분산 관련 GPU 메모리 (MB = 1024*1024 바이트)
	double denseMB = (double)nVox * 6 * sizeof(float) / (1024 * 1024);   // 기존 A 또는 B 하나 (밀집)
	double sliceMB = (double)sliceBytes / (1024 * 1024);
	double idxMB = (double)nVox * sizeof(int) / (1024 * 1024);
	double bandMB = (double)M_BAND * 6 * sizeof(float) / (1024 * 1024);
	double nowMB = 2 * sliceMB + idxMB + bandMB;
	printf("  띠 : %d / %zu 격자점 (%.2f %%)\n", M_BAND, nVox, 100.0 * M_BAND / nVox);
	printf("  공분산 GPU 메모리 : 기존 A + B = %.1f MB -> 장 버퍼 A, B 각 %.1f MB + 띠 번호 %.1f MB + 띠 배열 %.1f MB = %.1f MB (%.1f %%)\n",
		2 * denseMB, sliceMB, idxMB, bandMB, nowMB, 100.0 * nowMB / (2 * denseMB));
}

// [검증] 두 BMP 파일의 다른 픽셀 수. 파일이 없으면 -1
//  SaveBMP 는 1024 x 1024, 24비트, 줄 끝 패딩 0 -> 헤더 54바이트 뒤가 픽셀 3바이트씩
int CompareBMP(const char* fileA, const char* fileB) {
	std::ifstream a(fileA, std::ios::in | std::ios::binary);
	std::ifstream b(fileB, std::ios::in | std::ios::binary);
	if (!a.is_open() || !b.is_open()) return -1;
	std::vector<unsigned char> pa(54 + WIDTH * HEIGHT * 3), pb(54 + WIDTH * HEIGHT * 3);
	a.read((char*)pa.data(), pa.size());
	b.read((char*)pb.data(), pb.size());
	int diffPx = 0;
	for (int i = 0; i < WIDTH * HEIGHT; i++) {
		int o = 54 + i * 3;
		if (pa[o] != pb[o] || pa[o + 1] != pb[o + 1] || pa[o + 2] != pb[o + 2]) diffPx++;
	}
	return diffPx;
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
	size_t sliceBytes = (size_t)VOLX * VOLY * 6 * sizeof(float);   // (v1) 장 버퍼 하나
	GpuSetDim(VOLX, VOLY, VOLZ, BY_COUNT, BX_COUNT);
	CudaCheck(cudaGetLastError(), "크기 설정");
	CudaCheck(cudaMalloc(&d_vol, nVox), "d_vol 할당");
	CudaCheck(cudaMemcpy(d_vol, vol, nVox, cudaMemcpyHostToDevice), "d_vol 복사");
	CudaCheck(cudaMalloc(&d_bM, nBlk), "d_bM 할당");
	CudaCheck(cudaMemcpy(d_bM, bM, nBlk, cudaMemcpyHostToDevice), "d_bM 복사");
	CudaCheck(cudaMalloc(&d_Aslice, sliceBytes), "d_Aslice 할당");
	CudaCheck(cudaMalloc(&d_Bslice, sliceBytes), "d_Bslice 할당");
	Bslice = (float*)malloc(sliceBytes);
	CudaCheck(cudaMalloc(&d_img, WIDTH * HEIGHT * 3), "d_img 할당");
	std::cout << "GPU 메모리 : 장 버퍼 A, B 각 " << sliceBytes / (1024 * 1024) << " MB" << std::endl;
}

// (2차_4차) 해제 : MyInit 에서 잡은 것 전부
void MyFree() {
	CudaCheck(cudaFree(d_vol), "d_vol 해제");
	CudaCheck(cudaFree(d_bM), "d_bM 해제");
	CudaCheck(cudaFree(d_Aslice), "d_Aslice 해제");   // (v1)
	CudaCheck(cudaFree(d_Bslice), "d_Bslice 해제");   // (v1)
	CudaCheck(cudaFree(d_img), "d_img 해제");
	CudaCheck(cudaFree(d_idx), "d_idx 해제");         // (v1)
	CudaCheck(cudaFree(d_P), "d_P 해제");             // (v1)
	d_vol = 0; d_bM = 0; d_Aslice = 0; d_Bslice = 0; d_img = 0; d_idx = 0; d_P = 0;
	free(vol); free(bM); free(Bslice);
	vol = 0; bM = 0; Bslice = 0;
	std::cout << "해제 완료" << std::endl;   // [로그]
}

// 마지막 데이터의 사진을 창에 띄우기만 함 (렌더·저장은 main 반복에서)
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
	glutInit(&argc, argv); //GLUT 윈도우 함수
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutCreateWindow("OpenGL Drawing Example");
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
	glEnable(GL_TEXTURE_2D);
	printf("파라미터 : R = %d, sigma = %.2f, sn = %.2f, st = %.2f, sr = %.2f, rho %.2f ~ %.2f\n",   // [로그]
		R_KER, SIGMA, SIGMA_N, SIGMA_T, SIGMA_R, RHO_LO, RHO_HI);

	for (int f = 0; f < FILE_COUNT; f++) {
		MyInit(f);

		//==================== (v1) 전처리 : 장 단위 1단계 -> 2단계 -> 띠 고르기 ====================
		Preprocess();

		//==================== 렌더 + 저장 ====================
		Render();
		char outName[256];
		snprintf(outName, sizeof(outName), "%d_%s_slice_base.bmp", f + 1, OUT_NAME);
		SaveBMP(outName);

		MyFree();
	}

	glutDisplayFunc(MyDisplay);
	glutMainLoop();   // 창에는 마지막 데이터 사진 
	return 0;
}//
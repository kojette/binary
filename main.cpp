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
#include <cuda_runtime.h>   // v0 : cudaMalloc 등 (커널은 cu 파일)

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 800
#define WIDTH   512
#define HEIGHT  512
#define VOLX 256
#define VOLY 256	
#define VOLZ 225
const int BSIZE = 8;
const int BSHIFT = 3;

const int ISO = 120; //iso
const float ISO_LEVEL = 0.5f;

//수정A-1: 전역 파라미터로 관리(BSIZE로 나누되, 나머지 있음 +1)
const int BZ_COUNT = VOLZ / BSIZE + (VOLZ % BSIZE != 0); // 29
const int BY_COUNT = VOLY / BSIZE + (VOLY % BSIZE != 0); // 32
const int BX_COUNT = VOLX / BSIZE + (VOLX % BSIZE != 0); // 32

unsigned char ImageBuf[HEIGHT][WIDTH];
unsigned char MyTexture[HEIGHT][WIDTH][3];
unsigned char vol[VOLZ][VOLY][VOLX];


// 제거 : bm(블록 최소값). 이진에서는 "1이 하나라도 있나"만 보면 되므로
unsigned char bM[BZ_COUNT][BY_COUNT][BX_COUNT];

using namespace std;


//====================================================================
// v0 : CUDA 대표 파이프라인
//   전처리 1단계 : 등방 PCA (정육면체 창 R) -> 1차 공분산 A
//   전처리 2단계 : A 로 n0, rho -> 미끄럼틀로 sigma_t -> 팬케이크 PCA -> 2차 공분산 B
//   실시간       : 이진 바이섹션 교점 + 8이웃 B 를 (확신 w x 삼선형) 가중합 -> 최소 고유벡터
//====================================================================
// 1단계 파라미터 [사용자 결정 R=3]
const int   R_KER = 3;                  // 정육면체 창 반경
const float SIGMA = R_KER / 3.0f;       // 가우시안 폭 (3 sigma = R)
const float SPACING = 1.0f;             // 0.5 지점 수집 선 간격 (1 = 간선 중점)
// 2단계 파라미터
const float SIGMA_N = 0.6f;             // 팬케이크 법선 방향 폭 [문서 제안]
const float SIGMA_T_MAX = 4.0f;         // 팬케이크 접선 방향 최대 폭 [문서 제안]
const float RHO_FALLBACK = 0.7f;        // rho 이 이상이면 B = A [Claude 추천 임시]
const float RHO_SCALE = 0.3f;           // 미끄럼틀 s = 1/(1+(rho/0.3)^2) 의 0.3 [Claude 추천 임시]

// GPU 메모리 (Render 에서도 써야 하므로 전역)
unsigned char* d_vol = 0;   // 이진 볼륨
unsigned char* d_bM = 0;    // 블록 최대값
float* d_A = 0;             // 1차 공분산 6성분 (xx, yy, zz, xy, xz, yz), 띠 밖 = 0
float* d_B = 0;             // 2차 공분산 6성분, 띠 밖 = 0
unsigned char* d_img = 0;   // 렌더 결과 RGB


// ---- cu 파일의 창구 함수 선언 (커널 실행은 cu 쪽에서) ----
void GpuStep1(const unsigned char* d_vol, float* d_A, int z, int R, float sigma, float spacing);
void GpuStep2(const unsigned char* d_vol, const float* d_A, float* d_B, int z, float spacing,
	float sn, float stMax, float rhoFb, float rhoScale);
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

inline bool isOutside(const glm::vec3& p) {//범위 처리 따라, 알파 컬러에서는 불필요
	if (p.x >= VOLX - 1 || p.x < 0 ||//여기 -1로 처리함
		p.y >= VOLY - 1 || p.y < 0 ||
		p.z >= VOLZ - 1 || p.z < 0) return true;
	else
		return false;
}

int inline GetBlockId(glm::vec3 p) {//(개선+); 시프트 연산자로 블록 아이디 계산
	int x = p.x, y = p.y, z = p.z;
	int bx = x >> BSHIFT, by = y >> BSHIFT, bz = z >> BSHIFT;
	return (bx << 10) | (by << 5) | bz; // 수정A-4: 시프트 복호화로(어차피 진수표현만 상이)
}

//float화
float GetDensity(glm::vec3 p) {
	int ix = int(p.x); // 4.8 ->  4
	int iy = int(p.y); // 4.8 ->  4
	int iz = int(p.z); // 4.8 ->  4
	float wx = p.x - ix;
	float wy = p.y - iy;
	float wz = p.z - iz;
	float den = vol[iz][iy][ix] * (1 - wx) * (1 - wy) * (1 - wz)
		+ vol[iz][iy][ix + 1] * (wx) * (1 - wy) * (1 - wz)
		+ vol[iz][iy + 1][ix] * (1 - wx) * (wy) * (1 - wz)
		+ vol[iz][iy + 1][ix + 1] * (wx) * (wy) * (1 - wz)
		+ vol[iz + 1][iy][ix] * (1 - wx) * (1 - wy) * (wz)
		+vol[iz + 1][iy][ix + 1] * (wx) * (1 - wy) * (wz)
		+vol[iz + 1][iy + 1][ix] * (1 - wx) * (wy) * (wz)
		+vol[iz + 1][iy + 1][ix + 1] * (wx) * (wy) * (wz);
	return den;
}

// 부호장 : 안쪽이면 양수, 바깥이면 음수
//--------------------------------------------------------------------
// 변경 : 임계값 ISO(120) -> ISO_LEVEL(0.5)
//--------------------------------------------------------------------
inline float Phi(const glm::vec3& p) {
	if (isOutside(p)) return 0.0f - ISO_LEVEL;
	return GetDensity(p) - ISO_LEVEL;
}

//--------------------------------------------------------------------
// 보류 : 바이섹션. 이번 버전에서는 호출하지 않는다.
//--------------------------------------------------------------------
glm::vec3 Bisect(glm::vec3 a, glm::vec3 b) {   // a는 바깥, b는 안쪽
	for (int i = 0; i < 10; i++) {
		glm::vec3 m = (a + b) * 0.5f;//중점!
		if (Phi(m) < 0.0f) a = m;   // m이 바깥이면 a를 교체
		else               b = m;   // m이 안쪽이면 b를 교체
	}
	return (a + b) * 0.5f;
}

// 수정B-1: AABB 박스 체크 함수 분리~
inline bool AABB_box_check(const glm::vec3& RS, const glm::vec3& w, float& tm, float& tM) {
	float t1, t2;

	t1 = -RS.x / w.x;
	t2 = ((VOLX - 1) - RS.x) / w.x; // 수정A-3: 하드코딩제거(255-RS.x)->((VOLX-1)-RS.x)
	float xm = __min(t1, t2), xM = __max(t1, t2);
	t1 = -RS.y / w.y;
	t2 = ((VOLY - 1) - RS.y) / w.y;
	float ym = __min(t1, t2), yM = __max(t1, t2);
	t1 = -RS.z / w.z;
	t2 = ((VOLZ - 1) - RS.z) / w.z;
	float zm = __min(t1, t2), zM = __max(t1, t2);
	tm = __max(__max(xm, ym), zm);
	tM = __min(__min(xM, yM), zM);

	return tm < tM; // 교점 유효한거 있으면 참 리턴
}

// 수정B-2: 조명 연산 함수로 분리~
// 이진 볼륨 위에서는 중앙차분이 뭉텅뭉텅 꺾인 법선을 내놓는다. 그것이 출발점.
glm::vec3 lighting(const glm::vec3& p, const glm::vec3& rgb, const glm::vec3& w) {
	using namespace glm;
	//중앙차분법 기울기(노말) 계산
	float dx = (GetDensity(p + vec3(1, 0, 0)) - GetDensity(p - vec3(1, 0, 0))) * 0.5f;
	float dy = (GetDensity(p + vec3(0, 1, 0)) - GetDensity(p - vec3(0, 1, 0))) * 0.5f;
	float dz = (GetDensity(p + vec3(0, 0, 1)) - GetDensity(p - vec3(0, 0, 1))) * 0.5f;

	vec3 N(dx, dy, dz), V = -w, L = glm::normalize(-w + 0.3f * vec3(0, 1, 0));
	if (length(N) > 0.0f) N = normalize(N);

	vec3 H = normalize(L + V);

	float NL = fabs(dot(N, L));
	float NH = fabs(dot(N, H));

	float Ia = 0.25f, Id = 0.5f, Is = 0.9f;//살짝 밝게 // 합이 1인게 좋은데 여러 표현 가능
	vec3 Ka = rgb * 0.8f; //주변광 반사율 0.8 곱(어두운 배경 연출)
	vec3 Kd = rgb;
	vec3 Ks(1.2f, 0.8f, 0.8f); // 오팔 느낌

	vec3 I = Ia * Ka + Id * Kd * NL + Is * Ks * pow(NH, 30.0f);
	return clamp(I, 0.0f, 1.0f);
}

void SaveBMP(const char* filename) {//사진 저장(v1)
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

void Render(glm::vec3 eye) {
	using namespace glm;

	glm::vec3 at(128, 128, 112);
	glm::vec3 up(0, 1, 0);

	glm::vec3 w = glm::normalize(at - eye);
	glm::vec3 u = glm::normalize(glm::cross(up, w));
	glm::vec3 v = glm::normalize(glm::cross(w, u));

	auto start = std::chrono::high_resolution_clock::now();
	/////////////////레이캐스팅 : v0 GPU (픽셀 하나 = 스레드 하나)
	int nPix = WIDTH * HEIGHT;
	float e3[3] = { eye.x, eye.y, eye.z }, u3[3] = { u.x, u.y, u.z }, v3[3] = { v.x, v.y, v.z }, w3[3] = { w.x, w.y, w.z };
	GpuRender(d_vol, d_bM, d_B, d_img, e3, u3, v3, w3);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) std::cout << "[CUDA 에러] 렌더 커널 실행 : " << cudaGetErrorString(err) << std::endl;
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) std::cout << "[CUDA 에러] 렌더 커널 수행 : " << cudaGetErrorString(err) << std::endl;
	err = cudaMemcpy(&MyTexture[0][0][0], d_img, nPix * 3, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) std::cout << "[CUDA 에러] 이미지 복사 : " << cudaGetErrorString(err) << std::endl;

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

	//==================== v0 : GPU 메모리 준비 ====================
	cudaError_t err;
	size_t nVox = (size_t)VOLX * VOLY * VOLZ;
	size_t covBytes = nVox * 6 * sizeof(float);          // A, B 각각 약 354MB
	err = cudaMalloc(&d_vol, nVox);
	if (err != cudaSuccess) std::cout << "[CUDA 에러] d_vol 할당 : " << cudaGetErrorString(err) << std::endl;
	err = cudaMemcpy(d_vol, vol, nVox, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) std::cout << "[CUDA 에러] d_vol 복사 : " << cudaGetErrorString(err) << std::endl;
	err = cudaMalloc(&d_bM, sizeof(bM));
	if (err != cudaSuccess) std::cout << "[CUDA 에러] d_bM 할당 : " << cudaGetErrorString(err) << std::endl;
	err = cudaMemcpy(d_bM, bM, sizeof(bM), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) std::cout << "[CUDA 에러] d_bM 복사 : " << cudaGetErrorString(err) << std::endl;
	err = cudaMalloc(&d_A, covBytes);
	if (err != cudaSuccess) std::cout << "[CUDA 에러] d_A 할당 : " << cudaGetErrorString(err) << std::endl;
	err = cudaMemset(d_A, 0, covBytes);                   // 띠 밖 = 0
	if (err != cudaSuccess) std::cout << "[CUDA 에러] d_A 초기화 : " << cudaGetErrorString(err) << std::endl;
	err = cudaMalloc(&d_B, covBytes);
	if (err != cudaSuccess) std::cout << "[CUDA 에러] d_B 할당 : " << cudaGetErrorString(err) << std::endl;
	err = cudaMemset(d_B, 0, covBytes);
	if (err != cudaSuccess) std::cout << "[CUDA 에러] d_B 초기화 : " << cudaGetErrorString(err) << std::endl;
	err = cudaMalloc(&d_img, WIDTH * HEIGHT * 3);
	if (err != cudaSuccess) std::cout << "[CUDA 에러] d_img 할당 : " << cudaGetErrorString(err) << std::endl;
	std::cout << "GPU 메모리 : A, B 각 " << covBytes / (1024 * 1024) << " MB" << std::endl;

	//==================== 1단계 : 정육면체 창 등방 PCA -> A ====================
	auto pre1Start = std::chrono::high_resolution_clock::now();
	for (int z = 0; z < VOLZ; z++) {           // z 한 장씩 실행 (한 번에 너무 오래 돌면 윈도우가 GPU를 리셋함)
		GpuStep1(d_vol, d_A, z, R_KER, SIGMA, SPACING);
		err = cudaGetLastError();
		if (err != cudaSuccess) std::cout << "[CUDA 에러] 1단계 z=" << z << " : " << cudaGetErrorString(err) << std::endl;
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) std::cout << "[CUDA 에러] 1단계 수행 : " << cudaGetErrorString(err) << std::endl;
	auto pre1End = std::chrono::high_resolution_clock::now();
	std::cout << "1단계 (R=" << R_KER << ", s=" << SPACING << ") : "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(pre1End - pre1Start).count() << " ms" << std::endl;

	//==================== 2단계 : n0, rho -> 팬케이크 PCA -> B ====================
	auto pre2Start = std::chrono::high_resolution_clock::now();
	for (int z = 0; z < VOLZ; z++) {
		GpuStep2(d_vol, d_A, d_B, z, SPACING, SIGMA_N, SIGMA_T_MAX, RHO_FALLBACK, RHO_SCALE);
		err = cudaGetLastError();
		if (err != cudaSuccess) std::cout << "[CUDA 에러] 2단계 z=" << z << " : " << cudaGetErrorString(err) << std::endl;
		if (z % 10 == 0) {
			err = cudaDeviceSynchronize();
			if (err != cudaSuccess) std::cout << "[CUDA 에러] 2단계 수행 z=" << z << " : " << cudaGetErrorString(err) << std::endl;
			std::cout << "  2단계 z = " << z << " / " << VOLZ << std::endl;
		}
	}
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) std::cout << "[CUDA 에러] 2단계 수행 : " << cudaGetErrorString(err) << std::endl;
	auto pre2End = std::chrono::high_resolution_clock::now();
	std::cout << "2단계 (sn=" << SIGMA_N << ", st최대=" << SIGMA_T_MAX << ", rho폴백=" << RHO_FALLBACK
		<< ", rho척도=" << RHO_SCALE << ") : "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(pre2End - pre2Start).count() << " ms" << std::endl;
	//==================== 전처리 끝 ====================
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
	glEnable(GL_TEXTURE_2D);
}

void MyDisplay() {
	////////////////카메라 세팅
	static float t = 0;
	t += 1.0;
	glm::vec3 eye(0, 0, 100);   // 비교를 위해 고정iso
	//glm::vec3 eye(sin(t * 0.1) * 50, 0, 100);
	cout << glm::to_string(eye) << endl;

	Render(eye);
	static int saved = 0;                       // v0 : 사진은 처음 한 번만 저장
	if (saved == 0) {
		SaveBMP("v1-2_확신 가중 끔 (삼선형만).bmp");
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
	glutMainLoop();   // v0 : 창을 띄워 MyDisplay(렌더+저장)가 돌도록
	return 0;
}
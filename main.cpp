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
const float N_EPS = 1e-6f;   // nn이 이보다 작으면 법선 없음
using namespace std;
//영상 저장용
const char* SAVE_NAME = "step1.0_전처리(법선저장_상대좌표_정육면체R2_메모리압축없음_전처리에서Jacobi_최소고윳값법선_부호는무게중심반대)렌더링(N삼선형보간_부호무처리_바이섹션10_법선계산없음).bmp";
//---------- 공분산 전처리 (v3 추가) ----------
const int R = 2;  // 이웃 반경. 5x5x5 정육면체
//
//struct CovData {
//	float Sxx, Syy, Szz, Sxy, Sxz, Syz; // 누적합 (n으로 나누지 않음)
//	float Mx, My, Mz;                   // 무게중심용 좌표 합 (상대좌표)
//	float n;                            // 창 안의 1 개수
//};
//CovData covVol[VOLZ][VOLY][VOLX];
struct NormalData {
	float nx, ny, nz;   // 단위 법선. 경계가 아니거나 실패하면 (0,0,0)
};
NormalData normVol[VOLZ][VOLY][VOLX];//최근접 법선용. 

//---------- 대칭 3x3 고유분해 : Jacobi 회전법 (v3-2 추가) ----------
// A는 파괴됨. eval[i] 와 evec의 i번째 "열"이 짝.
void Jacobi3(double A[3][3], double eval[3], double evec[3][3]) {
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++) evec[i][j] = (i == j) ? 1.0 : 0.0;

	for (int sweep = 0; sweep < 20; sweep++) {
		double off = fabs(A[0][1]) + fabs(A[0][2]) + fabs(A[1][2]);
		if (off < 1e-12) break;

		for (int p = 0; p < 2; p++)
			for (int q = p + 1; q < 3; q++) {
				if (fabs(A[p][q]) < 1e-15) continue;

				// 비대각 원소 하나를 정확히 0으로 만드는 회전각
				double theta = (A[q][q] - A[p][p]) / (2.0 * A[p][q]);
				double t = (theta >= 0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
				double c = 1.0 / sqrt(t * t + 1.0);
				double s = t * c;

				for (int k = 0; k < 3; k++) {        // A * J
					double akp = A[k][p], akq = A[k][q];
					A[k][p] = c * akp - s * akq;
					A[k][q] = s * akp + c * akq;
				}
				for (int k = 0; k < 3; k++) {        // J^T * A
					double apk = A[p][k], aqk = A[q][k];
					A[p][k] = c * apk - s * aqk;
					A[q][k] = s * apk + c * aqk;
				}
				for (int k = 0; k < 3; k++) {        // V * J
					double vkp = evec[k][p], vkq = evec[k][q];
					evec[k][p] = c * vkp - s * vkq;
					evec[k][q] = s * vkp + c * vkq;
				}
			}
	}
	eval[0] = A[0][0]; eval[1] = A[1][1]; eval[2] = A[2][2];
}

void SaveBMP(const char* filename) {
	int rowSize = WIDTH * 3;
	int pad = (4 - (rowSize & 3)) & 3;   // 0~3
	int dataSize = (rowSize + pad) * HEIGHT;
	int fileSize = 54 + dataSize;

	unsigned char header[54] = { 0 };
	header[0] = 'B'; header[1] = 'M';
	header[2] = fileSize; header[3] = fileSize >> 8;
	header[4] = fileSize >> 16; header[5] = fileSize >> 24;
	header[10] = 54;                       // 픽셀 데이터 시작 오프셋
	header[14] = 40;                       // DIB 헤더 크기
	header[18] = WIDTH; header[19] = WIDTH >> 8;
	header[20] = WIDTH >> 16; header[21] = WIDTH >> 24;
	header[22] = HEIGHT; header[23] = HEIGHT >> 8;
	header[24] = HEIGHT >> 16; header[25] = HEIGHT >> 24;
	header[26] = 1;                        // 플레인 수
	header[28] = 24;                       // 픽셀당 비트
	header[34] = dataSize; header[35] = dataSize >> 8;
	header[36] = dataSize >> 16; header[37] = dataSize >> 24;

	std::ofstream f(filename, std::ios::out | std::ios::binary);
	if (!f.is_open()) {
		std::cout << "save error : " << filename << std::endl;
		return;
	}
	f.write((char*)header, 54);

	unsigned char padding[3] = { 0, 0, 0 };
	for (int y = 0; y < HEIGHT; y++) {     // MyTexture의 y=0이 아래줄
		for (int x = 0; x < WIDTH; x++) {
			unsigned char bgr[3];
			bgr[0] = MyTexture[y][x][2];   // B
			bgr[1] = MyTexture[y][x][1];   // G
			bgr[2] = MyTexture[y][x][0];   // R
			f.write((char*)bgr, 3);
		}
		f.write((char*)padding, pad);
	}
	f.close();
	std::cout << "saved : " << filename << std::endl;
}
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
	////---------- v4 : 누적합 보간 -> 공분산 복원 -> Jacobi -> 법선 ----------
	//int ix = int(p.x);
	//int iy = int(p.y);
	//int iz = int(p.z);
	//float wx = p.x - ix;
	//float wy = p.y - iy;
	//float wz = p.z - iz;

	//// 보간 결과를 담을 그릇. 0에서 시작해 8개 모서리를 더한다.
	//float sxx = 0, syy = 0, szz = 0, sxy = 0, sxz = 0, syz = 0;
	//float mx = 0, my = 0, mz = 0;
	//float nn = 0;

	//for (int dz = 0; dz < 2; dz++)
	//	for (int dy = 0; dy < 2; dy++)
	//		for (int dx = 0; dx < 2; dx++) {
	//			float wgt = (dx ? wx : 1.0f - wx)
	//				* (dy ? wy : 1.0f - wy)
	//				* (dz ? wz : 1.0f - wz);

	//			const CovData& cd = covVol[iz + dz][iy + dy][ix + dx];

	//			sxx += wgt * cd.Sxx;  syy += wgt * cd.Syy;  szz += wgt * cd.Szz;
	//			sxy += wgt * cd.Sxy;  sxz += wgt * cd.Sxz;  syz += wgt * cd.Syz;
	//			mx += wgt * cd.Mx;  my += wgt * cd.My;  mz += wgt * cd.Mz;
	//			nn += wgt * cd.n;
	//		}

	//vec3 N(0.0f);   // 8이웃이 전부 경계가 아니면 0벡터로 남는다

	//if (nn >= N_EPS) {
	//	double inv = 1.0 / nn;
	//	double C[3][3];
	//	C[0][0] = sxx * inv - (mx * inv) * (mx * inv);
	//	C[1][1] = syy * inv - (my * inv) * (my * inv);
	//	C[2][2] = szz * inv - (mz * inv) * (mz * inv);
	//	C[0][1] = C[1][0] = sxy * inv - (mx * inv) * (my * inv);
	//	C[0][2] = C[2][0] = sxz * inv - (mx * inv) * (mz * inv);
	//	C[1][2] = C[2][1] = syz * inv - (my * inv) * (mz * inv);

	//	double eval[3], evec[3][3];
	//	Jacobi3(C, eval, evec);

	//	// 가장 작은 고윳값의 고유벡터 = 법선
	//	int k = 0;
	//	if (eval[1] < eval[k]) k = 1;
	//	if (eval[2] < eval[k]) k = 2;
	//	double nx = evec[0][k], ny = evec[1][k], nz = evec[2][k];

	//	double len = sqrt(nx * nx + ny * ny + nz * nz);
	//	if (len > 1e-12) {
	//		nx /= len; ny /= len; nz /= len;
	//		// 부호 : 무게중심의 반대쪽이 바깥
	//		if (nx * (-mx) + ny * (-my) + nz * (-mz) < 0.0) {
	//			nx = -nx; ny = -ny; nz = -nz;
	//		}
	//		N = vec3((float)nx, (float)ny, (float)nz);
	//	}
	//}
	//---------- v4 끝 ----------
	////---------- v6 : 보간 없음. 가장 가까운 격자점의 법선을 그대로 읽는다 ----------
	//int ix = int(p.x + 0.5f);   // 반올림
	//int iy = int(p.y + 0.5f);
	//int iz = int(p.z + 0.5f);

	//if (ix < 0) ix = 0;  if (ix > VOLX - 1) ix = VOLX - 1;
	//if (iy < 0) iy = 0;  if (iy > VOLY - 1) iy = VOLY - 1;
	//if (iz < 0) iz = 0;  if (iz > VOLZ - 1) iz = VOLZ - 1;

	//const NormalData& nd = normVol[iz][iy][ix];
	//vec3 N(nd.nx, nd.ny, nd.nz);
	////---------- v6 끝 ----------
	//---------- v7 : 8이웃 법선 삼선형 보간. 부호 처리 없음 ----------
	int ix = int(p.x);          // 내림. 8이웃의 기준 모서리
	int iy = int(p.y);
	int iz = int(p.z);
	float wx = p.x - ix;
	float wy = p.y - iy;
	float wz = p.z - iz;

	vec3 N(0.0f);

	for (int dz = 0; dz < 2; dz++)
		for (int dy = 0; dy < 2; dy++)
			for (int dx = 0; dx < 2; dx++) {
				float wgt = (dx ? wx : 1.0f - wx)
					* (dy ? wy : 1.0f - wy)
					* (dz ? wz : 1.0f - wz);

				const NormalData& nd = normVol[iz + dz][iy + dy][ix + dx];
				N += wgt * vec3(nd.nx, nd.ny, nd.nz);
			}
	//---------- v7 끝 ----------

	//vec3 N(dx, dy, dz), V = -w, L = glm::normalize(-w + 0.3f * vec3(0, 1, 0));
	vec3 V = -w, L = glm::normalize(-w + 0.3f * vec3(0, 1, 0));
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

void Render(glm::vec3 eye) {
	using namespace glm;

	glm::vec3 at(128, 128, 112);
	glm::vec3 up(0, 1, 0);

	glm::vec3 w = glm::normalize(at - eye);
	glm::vec3 u = glm::normalize(glm::cross(up, w));
	glm::vec3 v = glm::normalize(glm::cross(w, u));

	auto start = std::chrono::high_resolution_clock::now();
	const float supersampling = 0.5;
	/////////////////레이캐스팅
	for (int y = 0; y < HEIGHT; y++) { // 영상의 y좌표
		for (int x = 0; x < WIDTH; x++) { // 영상의 x좌표
			glm::vec3 RS = eye + u * (x - WIDTH * 0.5f) * supersampling + v * (y - HEIGHT * 0.5f) * supersampling;

			float tm, tM; // 수정B-1-(2): AABB 박스 체크 함수 분리~
			if (!AABB_box_check(RS, w, tm, tM)) continue; // 박스로 광선 가는거 아니면 패스 

			glm::vec3 col(0.0f);
			const float step = 0.5f; // "자잘수정1": float의 경우, f를 추가해야 유리

			float tBefore = tm;//iso4; 직전 샘플임을 보장하기 위한
			float phiBefore = Phi(RS + w * tm);
			for (float t = tm; t < tM; t = t + step) { // 광선을 진행하자
				glm::vec3 p = RS + w * t;
				if (isOutside(p))
					continue;

				// 내(p)가 속한 블록의 max 안다고 가정.
				int bid = GetBlockId(p); // 123456

				// 수정A-5: 비트연산자 활용해 봄. 
				int bz = bid & 0x1F; //1F(16+15)임 즉, 11111이고 &연산함.              
				int by = (bid >> 5) & 0x1F; // 5개 지우고 남은 오른쪽 5개 추출
				int bx = (bid >> 10) & 0x1F; // 이하 동일

				//--------------------------------------------------------
				// 변경 : bM < ISO -> bM == 0
				//        vol 이 0/1 이므로 블록 최대값이 0이면 통째로 빈 블록.
				//--------------------------------------------------------
				if (bM[bz][by][bx] == 0) {
					float jump = 0;
					int nextBid;
					// 빈 블록이니까, 연산을 건너뛰자. 광선을 빠르게 전진하자.
					do {
						jump += 1.0f;
						nextBid = GetBlockId(p + w * jump); // 추가 전진
					} while (bid == nextBid);
					t = t + (jump - step);
					tBefore = t;//iso4
					phiBefore = Phi(RS + w * t);
					continue;
				}

				float phi = Phi(p);   // = GetDensity(p) - ISO_LEVEL

				if (phiBefore * phi < 0.0f) { //부호 반전 검출
					//glm::vec3 hit = p;
					glm::vec3 pBefore = RS + w * tBefore;//바이섹션 추가~
					glm::vec3 hit = (phiBefore < 0.0f) ? Bisect(pBefore, p)
						: Bisect(p, pBefore);

					glm::vec3 rgb(0.9f, 0.85f, 0.8f);
					col = lighting(hit, rgb, w);
					break;
				}

				tBefore = t;
				phiBefore = phi;
			}
			MyTexture[y][x][0] = int(col.r * 255);
			MyTexture[y][x][1] = int(col.g * 255);
			MyTexture[y][x][2] = int(col.b * 255);
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "실행 시간: " << duration.count() * 0.001f << " ms" << std::endl;
	glTexImage2D(GL_TEXTURE_2D, 0, 3, WIDTH, HEIGHT, 0, GL_RGB,
		GL_UNSIGNED_BYTE, &MyTexture[0][0][0]);
}

void MyInit() {
	glClearColor(0.0, 0.0, 0.0, 0.0);
	FileRead();


	// 이진화 전처리. 반드시 GenBlocks() 보다 먼저
	for (int z = 0; z < VOLZ; z++)
		for (int y = 0; y < VOLY; y++)
			for (int x = 0; x < VOLX; x++)
				vol[z][y][x] = (vol[z][y][x] >= ISO);
	//--------------------------------------------------------------------

	auto preStart = std::chrono::high_resolution_clock::now();//측정용 시간
	//---------------------------------------전처리 시작

	for (int z = 0; z < VOLZ; z++)
		for (int y = 0; y < VOLY; y++)
			for (int x = 0; x < VOLX; x++) {

				//covVol[z][y][x].n = 0.0f;  // 기본은 비어있음 표시

				// --- 경계 판정: 6-이웃 중 자신과 다른 값이 하나라도 있으면 경계 (0쪽 1쪽 모두) ---
				if (x == 0 || y == 0 || z == 0 ||
					x == VOLX - 1 || y == VOLY - 1 || z == VOLZ - 1) continue;

				unsigned char c = vol[z][y][x];
				bool isBoundary =
					(vol[z][y][x - 1] != c) || (vol[z][y][x + 1] != c) ||
					(vol[z][y - 1][x] != c) || (vol[z][y + 1][x] != c) ||
					(vol[z - 1][y][x] != c) || (vol[z + 1][y][x] != c);
				if (!isBoundary) continue;

				// --- 5x5x5 창에서 값이 1인 복셀의 상대좌표!!를 누적 ---
				float Sxx = 0, Syy = 0, Szz = 0, Sxy = 0, Sxz = 0, Syz = 0;
				float Mx = 0, My = 0, Mz = 0;
				float n = 0;

				for (int dz = -R; dz <= R; dz++)
					for (int dy = -R; dy <= R; dy++)
						for (int dx = -R; dx <= R; dx++) {
							//if (dx * dx + dy * dy + dz * dz > R * R) continue;
							int nx = x + dx, ny = y + dy, nz = z + dz;
							if (nx < 0 || ny < 0 || nz < 0 ||
								nx >= VOLX || ny >= VOLY || nz >= VOLZ) continue;
							if (vol[nz][ny][nx] == 0) continue;

							float fx = (float)dx, fy = (float)dy, fz = (float)dz;
							Mx += fx;  My += fy;  Mz += fz;
							Sxx += fx * fx;  Syy += fy * fy;  Szz += fz * fz;
							Sxy += fx * fy;  Sxz += fx * fz;  Syz += fy * fz;
							n += 1.0f;
						}

				/*covVol[z][y][x].Mx = Mx;   covVol[z][y][x].My = My;   covVol[z][y][x].Mz = Mz;
				covVol[z][y][x].Sxx = Sxx; covVol[z][y][x].Syy = Syy; covVol[z][y][x].Szz = Szz;
				covVol[z][y][x].Sxy = Sxy; covVol[z][y][x].Sxz = Sxz; covVol[z][y][x].Syz = Syz;
				covVol[z][y][x].n = n;*/
				//---------- v6 : 여기서 바로 공분산 복원 -> Jacobi -> 법선 ----------
				if (n < N_EPS) continue;   // 창이 비었으면 법선 없음 (0,0,0) 유지

				double inv = 1.0 / n;
				double C[3][3];
				C[0][0] = Sxx * inv - (Mx * inv) * (Mx * inv);
				C[1][1] = Syy * inv - (My * inv) * (My * inv);
				C[2][2] = Szz * inv - (Mz * inv) * (Mz * inv);
				C[0][1] = C[1][0] = Sxy * inv - (Mx * inv) * (My * inv);
				C[0][2] = C[2][0] = Sxz * inv - (Mx * inv) * (Mz * inv);
				C[1][2] = C[2][1] = Syz * inv - (My * inv) * (Mz * inv);

				double eval[3], evec[3][3];
				Jacobi3(C, eval, evec);

				// 가장 작은 고윳값의 고유벡터 = 법선
				int k = 0;
				if (eval[1] < eval[k]) k = 1;
				if (eval[2] < eval[k]) k = 2;
				double nx = evec[0][k], ny = evec[1][k], nz = evec[2][k];

				double len = sqrt(nx * nx + ny * ny + nz * nz);
				if (len < 1e-12) continue;   // 실패. (0,0,0) 유지
				nx /= len; ny /= len; nz /= len;

				// 부호 : 무게중심의 반대쪽이 바깥
				if (nx * (-Mx) + ny * (-My) + nz * (-Mz) < 0.0) {
					nx = -nx; ny = -ny; nz = -nz;
				}

				normVol[z][y][x].nx = (float)nx;
				normVol[z][y][x].ny = (float)ny;
				normVol[z][y][x].nz = (float)nz;
			}
	//---------------------------------------전처리 끝

	auto preEnd = std::chrono::high_resolution_clock::now();
	auto preDur = std::chrono::duration_cast<std::chrono::microseconds>(preEnd - preStart);
	std::cout << "전처리(공분산) 시간: " << preDur.count() * 0.001f << " ms" << std::endl;

	GenBlocks(); // 파일은 읽고 난 다음에.
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
	SaveBMP(SAVE_NAME);//영상 저장용
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
	glutMainLoop();
	return 0;
}
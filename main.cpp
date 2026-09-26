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

using namespace std;

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
					//------------------------------------------------
					// 변경 : 바이섹션 미사용. 현재 샘플점을 교점으로 삼는다.
					//------------------------------------------------
					glm::vec3 hit = p;

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


	// 이진화 전처리. 반드시 GenBlocks() 보다 먼저 와야 한다.
	for (int z = 0; z < VOLZ; z++)
		for (int y = 0; y < VOLY; y++)
			for (int x = 0; x < VOLX; x++)
				vol[z][y][x] = (vol[z][y][x] >= ISO);
	printf("binarized (ISO = %d, inside : d >= ISO)\n", ISO);
	//--------------------------------------------------------------------

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
	//glutMainLoop();
	return 0;
}
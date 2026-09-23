// =====================================================================
//  main.cpp  -  v1 ~ v14 일괄 실행 (전처리 시간 / 실행 시간 / BMP 저장)
//
//  그냥 실행하면 v1 -> v14 순서로 한 버전씩 돌고, 버전마다
//      1) 전처리 시간   2) 실행(렌더링) 시간   3) BMP 저장 (v번호_세부내용.bmp)
//  을 콘솔에 찍은 뒤 줄바꿈을 두 번 한다. 로그 파일은 만들지 않는다.
//  작업 디렉터리에 bighead.den 이 있어야 하고, BMP 도 작업 디렉터리에 저장된다.
//  돌리는 동안 GLUT 창은 멈춘 것처럼 보인다(메시지 루프를 안 돌리므로 정상). 모두 끝나면 Enter 를 기다린다.
//
//  각 버전 = 깃허브(kojette/binary) 커밋의 main.cpp 를 namespace 로 감싼 것.
//  원본과 달라진 곳은 아래 넷뿐이고 전부 "[harness]" 로 표시해 두었다.
//    (a) SAVE_NAME : 파일명을 v번호_세부내용 으로 통일 (step 은 코드의 값인 0.5 로 적음)
//    (b) 격자 크기 거대 배열(covVol / normVol / tenVol) : 전역 배열 -> 포인터
//        14개 버전의 전역 배열을 한 exe 에 다 두면 정적 데이터가 2GB 를 넘어 링크가 안 된다.
//        그래서 버전마다 새로 잡고 끝나면 반납한다. 새로 잡은 페이지는 0 으로 채워져 있고 첫 접촉 때
//        페이지 폴트가 나므로, 원본 전역 배열(BSS)과 같은 조건이다. 접근 문법은 그대로다.
//    (c) v1 에만 SaveBMP 를 붙임 (원본 v1 에는 저장 기능이 없다)
//    (d) main() : 창 하나 만들고, 각 버전의 MyInit() -> MyDisplay() 를 한 번씩 직접 호출
//        (원본은 glutMainLoop 안에서 MyDisplay 가 불렸다. 1프레임만 재는 것은 같다)
//
//  버전 <-> 커밋
//    v1   1c2ecde  "Replace repo contents with main.cpp"
//           main_v1_이진화. USE_BINARY 스위치, 중앙차분 법선. 원본엔 BMP 저장이 없어 v2 의 SaveBMP 를 그대로 붙임
//    v2   ca18065  "add capture"
//           v2 이진화 정리 + BMP 캡처("베이스"). 4769172(code cleaning)는 캡처만 없는 동일 코드라 생략
//    v3   54b0947  "전처리 공분산저장_상대좌표_정육면체2_메모리압축안함_정규화안함"
//           공분산 누적합 전처리만 추가. 렌더링은 아직 중앙차분. 원본은 SaveBMP 호출이 주석이라 저장을 여기서 함
//    v4   244eab0  "Jacobi_기초렌더링"
//           누적합 보간 -> 공분산 복원 -> Jacobi. 이 커밋의 R 은 4 (커밋 그대로)
//    v5   5ce98c7  "Bisect추가_R실험_구커널시도"
//           바이섹션 추가, R=2. 92a75c4(step 0.5 고정)는 파일명만 다른 동일 코드라 생략
//    v6   e9d2916  "전처리 법선 저장 + 실시간 최근접 N"
//           전처리에서 법선 저장, 렌더링은 최근접 N
//    v7   cb3a646  "8이웃 삼선형 N 보간"
//           N 삼선형 보간, 부호 무처리
//    v8   2ed55fe  "8이웃 삼선형 보간 최근접 기분 부호 정렬"
//           중심(최근접) 기준 부호 정렬
//    v9   ef54af1  "8이웃삼선형법선보간_중심최근접기준부호정렬"
//           코드는 시선 기준 정렬(v9). 커밋 메시지는 v8 것이 남아 있음. b5f3856 은 파일명만 다른 동일 코드라 생략
//    v10  d4c7b63  "NNT텐서보간"
//           N 에서 NNT 를 만들어 보간 -> 최대 고유벡터
//    v11  6486ce6  "부호장 기준 정렬"
//           부호장 기울기 기준 정렬
//    v12  7f45039  "v12:실시간"
//           저장 없이 교점마다 창을 훑어 PCA. 전처리 시간은 0 ms 로 찍힘
//    v13  078f95b  "v13 공분산을 전처리로 저장"
//           공분산 6개 저장, 픽셀마다 Jacobi
//    v14  1d679df  "v14 텐서 저장 at 전처리"
//           NNT 텐서 6개 저장, 픽셀마다 Jacobi
//  생략한 커밋 : 80d1139(다른 프로그램), 4769172, aa2b0c2(v3-2 디버그), 28f659b, 92a75c4, b5f3856
// =====================================================================
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <GL/glut.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
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

// ---------------------------------------------------------------------
// [harness] 거대 배열 할당 / 반납. 0 으로 채워진 새 페이지를 그때그때 받는다.
//   Windows : VirtualAlloc (필요할 때 페이지가 붙는다. 원본 BSS 와 동일한 동작)
//   그 외    : calloc
// ---------------------------------------------------------------------
#ifdef _WIN32
static void* RawAlloc(size_t bytes) { return VirtualAlloc(NULL, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE); }
static void  BigFree(void* p) { if (p) VirtualFree(p, 0, MEM_RELEASE); }
#else
static void* RawAlloc(size_t bytes) { return calloc(1, bytes); }
static void  BigFree(void* p) { free(p); }
#endif
static void* BigAlloc(size_t bytes) {
	void* p = RawAlloc(bytes);
	if (!p) { std::cout << "메모리 할당 실패 : " << (bytes >> 20) << " MB" << std::endl; exit(1); }
	return p;
}


// =====================================================================
//  v1   커밋 1c2ecde  "Replace repo contents with main.cpp"
// =====================================================================
namespace v1 {

	// 시간 측정등 고성능 함수

	// =====================================================================
	//  main_v1_이진화.cpp
	//  - 베이스 : 아이소서페이스 레이캐스팅 코드
	//  - 변경   : 알파/컬러 테이블 전면 제거, 이진(binary) 볼륨 처리부 추가
	//  - 관례   : density >= ISO 이면 안쪽(inside, 가시), < ISO 이면 바깥
	//             => 부호장 phi >= 0 이면 안쪽. 경계값 120 자체는 안쪽에 포함.
	//  - 모드   : USE_BINARY 스위치로 (A)원본밀도 / (B)이진볼륨 을 전환
	//             (A) 보간 후 이진화 : 원본 밀도를 삼선형 보간 -> 120 비교
	//             (B) 이진화 후 보간 : 복셀을 0/1 로 만든 뒤 그 장(場)을 보간
	//  - 미구현 : 바이섹션(정밀 교점 탐색)은 함수만 남기고 호출하지 않음
	// =====================================================================

	const int BSIZE = 8;
	const int BSHIFT = 3;

	const int ISO = 120; //iso

	// [추가] 렌더링 모드 스위치
	//   true  : 이진 볼륨(0/1)을 보간해서 렌더링   -> 표면 등위값 0.5
	//   false : 원본 밀도를 보간해서 렌더링(비교군) -> 표면 등위값 120
	const bool USE_BINARY = true;

	// [추가] 모드에 따른 표면 등위값(iso-level)
	const float ISO_LEVEL = USE_BINARY ? 0.5f : float(ISO);

	//수정A-1: 전역 파라미터로 관리(BSIZE로 나누되, 나머지 있음 +1)
	const int BZ_COUNT = VOLZ / BSIZE + (VOLZ % BSIZE != 0); // 29
	const int BY_COUNT = VOLY / BSIZE + (VOLY % BSIZE != 0); // 32
	const int BX_COUNT = VOLX / BSIZE + (VOLX % BSIZE != 0); // 32

	unsigned char ImageBuf[HEIGHT][WIDTH];
	unsigned char MyTexture[HEIGHT][WIDTH][3];
	unsigned char vol[VOLZ][VOLY][VOLX];

	// [추가] 이진 볼륨. 0 또는 1 만 담긴다.
	unsigned char bin[VOLZ][VOLY][VOLX];

	//수정A-1-(2):전역 파라미터 수정했으니까 이걸로 쓰면 좋을듯.
	// [제거] bm(블록 최소값) 삭제 - 이진/아이소 스킵에는 최대값만 있으면 충분.
	//        bin[z][y][x]==1 인 복셀이 있다  <=>  bM >= ISO 이므로
	//        bM 하나로 두 모드 모두 커버된다.
	unsigned char bM[BZ_COUNT][BY_COUNT][BX_COUNT];

	// [제거] alphaTable / sumTable / colorTableR,G,B 전역 배열 삭제
	// [제거] struct alphaPoint, class AlphaTable, struct colorPoint, class ColorTable 삭제
	// [제거] InitTables() 삭제

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

	// [추가] 이진화. 관례: d >= ISO 이면 안쪽(1), 그 미만이면 바깥(0).
	void Binarize() {
		for (int z = 0; z < VOLZ; z++)
			for (int y = 0; y < VOLY; y++)
				for (int x = 0; x < VOLX; x++)
					bin[z][y][x] = (vol[z][y][x] >= ISO);
		printf("binarized (ISO = %d, inside : d >= ISO)\n", ISO);
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

	// (A) 원본 밀도 삼선형 보간 : 비교군으로 남겨둔다.
	float GetDensity(glm::vec3 p) {
		int ix = int(p.x); // 4.8 ->  4
		int iy = int(p.y); // 4.8 ->  4
		int iz = int(p.z); // 4.8 ->  4
		float wx = p.x - ix;
		float wy = p.y - iy;
		float wz = p.z - iz;
		// linear interpolation : 직선형 보간
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

	// [추가] (B) 이진 볼륨 삼선형 보간. 반환값은 0.0 ~ 1.0 실수.
	//        원본의 계조가 사라진 자리에 계단과 톱니가 남는다. 연구의 출발점.
	float GetBinary(glm::vec3 p) {
		int ix = int(p.x);
		int iy = int(p.y);
		int iz = int(p.z);
		float wx = p.x - ix;
		float wy = p.y - iy;
		float wz = p.z - iz;
		float d = bin[iz][iy][ix] * (1 - wx) * (1 - wy) * (1 - wz)
			+ bin[iz][iy][ix + 1] * (wx) * (1 - wy) * (1 - wz)
			+ bin[iz][iy + 1][ix] * (1 - wx) * (wy) * (1 - wz)
			+ bin[iz][iy + 1][ix + 1] * (wx) * (wy) * (1 - wz)
			+ bin[iz + 1][iy][ix] * (1 - wx) * (1 - wy) * (wz)
			+bin[iz + 1][iy][ix + 1] * (wx) * (1 - wy) * (wz)
			+bin[iz + 1][iy + 1][ix] * (1 - wx) * (wy) * (wz)
			+bin[iz + 1][iy + 1][ix + 1] * (wx) * (wy) * (wz);
		return d;
	}

	// [추가] 모드 분기 지점. 여기 하나만 갈아끼우면 전체가 따라온다.
	inline float GetField(const glm::vec3& p) {
		return USE_BINARY ? GetBinary(p) : GetDensity(p);
	}

	// 부호장 : 안쪽이면 양수(>=0), 바깥이면 음수
	inline float Phi(const glm::vec3& p) {
		if (isOutside(p)) return 0.0f - ISO_LEVEL;
		return GetField(p) - ISO_LEVEL;
	}

	// [보류] 바이섹션. 이번 버전에서는 호출하지 않는다.
	//        다음 버전(_바이섹션추가)에서 켠 뒤, 이 버전과 화질을 비교할 것.
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
	// [변경] GetDensity -> GetField. 이진 모드에서는 중앙차분이 -1,0,+1 세 값만
	//        내놓으므로 법선이 여섯 방향으로 뭉텅뭉텅 꺾인다. 그것이 문제 제기 그림.
	glm::vec3 lighting(const glm::vec3& p, const glm::vec3& rgb, const glm::vec3& w) {
		using namespace glm;
		//중앙차분법 기울기(노말) 계산
		float dx = (GetField(p + vec3(1, 0, 0)) - GetField(p - vec3(1, 0, 0))) * 0.5f;
		float dy = (GetField(p + vec3(0, 1, 0)) - GetField(p - vec3(0, 1, 0))) * 0.5f;
		float dz = (GetField(p + vec3(0, 0, 1)) - GetField(p - vec3(0, 0, 1))) * 0.5f;

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

				// [변경] 누적(r_sum,g_sum,b_sum,a_sum) 제거. 첫 교차에서 끝난다.
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

					// 블록 안에 임계 넘는 값이 하나도 없다 = 이진 모드에선 전부 0인 블록
					if (bM[bz][by][bx] < ISO) {
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

					float phi = Phi(p);   // = GetField(p) - ISO_LEVEL

					if (phiBefore * phi < 0.0f) { //부호 반전 검출
						// [변경] 바이섹션 미사용. 현재 샘플점을 그대로 교점으로 삼는다.
						//        다음 버전에서 Bisect 를 켜고 이 결과와 비교할 것.
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
		Binarize();  // [추가] 파일 읽은 직후 이진화
		GenBlocks(); // 파일은 읽고 난 다음에.
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
		glEnable(GL_TEXTURE_2D);
		// [제거] InitTables(); 호출 삭제
		printf("mode : %s\n", USE_BINARY ? "BINARY (interp of 0/1)" : "DENSITY (interp of raw)");
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

	// ===================== [harness] 여기부터는 원본 커밋에 없는 일괄 실행용 코드 =====================
	// v1 원본에는 BMP 저장이 없다. v2(ca18065)의 SaveBMP 를 글자 그대로 옮겨 붙였다. 렌더링 코드는 건드리지 않았다.
	const char* SAVE_NAME = "v1_step0.5_전처리(이진화만_별도bin배열_USE_BINARY스위치)렌더링(이진값삼선형보간_중앙차분법선_바이섹션없음_교점은샘플점_법선계산은lighting내부).bmp";
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
	void HarnessAlloc() {}   // 이 버전은 거대 배열이 없다
	void HarnessFree() {}
	void HarnessAfterDisplay() { /*SaveBMP(SAVE_NAME);*/ }   // 원본 MyDisplay 는 저장을 안 하므로 여기서 저장
} // namespace v1

// =====================================================================
//  v2   커밋 ca18065  "add capture"
// =====================================================================
namespace v2 {

	// 시간 측정등 고성능 함수

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
	//영상 저장용
	const char* SAVE_NAME = "v2_베이스_step0.5_전처리(이진화만_vol덮어쓰기)렌더링(이진값삼선형보간_중앙차분법선_바이섹션없음_교점은샘플점_법선계산은lighting내부).bmp"; // [harness] 파일명을 v번호_세부내용 으로 통일  
	// 추가 : 렌더 결과(MyTexture)를 24비트 BMP로 저장.
	//        BMP는 아래->위, BGR 순서로 저장한다.
	//        WIDTH*3 이 4의 배수가 아닐 경우를 대비해 패딩을 넣는다.
	//--------------------------------------------------------------------
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
		/*SaveBMP(SAVE_NAME);*///영상 저장용
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

	// ===================== [harness] 여기부터는 원본 커밋에 없는 일괄 실행용 코드 =====================
	void HarnessAlloc() {}   // 이 버전은 거대 배열이 없다
	void HarnessFree() {}
	void HarnessAfterDisplay() {}   // 원본 MyDisplay 가 이미 SaveBMP(SAVE_NAME) 를 한다
} // namespace v2

// =====================================================================
//  v3   커밋 54b0947  "전처리 공분산저장_상대좌표_정육면체2_메모리압축안함_정규화안함"
// =====================================================================
namespace v3 {

	// 시간 측정등 고성능 함수

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

	//---------- 공분산 전처리 (v3 추가) ----------
	const int R = 2;  // 이웃 반경. 5x5x5 정육면체

	struct CovData {
		float Sxx, Syy, Szz, Sxy, Sxz, Syz; // 누적합 (n으로 나누지 않음)
		float Mx, My, Mz;                   // 무게중심용 좌표 합 (상대좌표)
		float n;                            // 창 안의 1 개수
	};
	CovData(*covVol)[VOLY][VOLX] = nullptr;   // [harness] 원본: CovData covVol[VOLZ][VOLY][VOLX];
	//--------------------------------------------------


	//영상 저장용
	const char* SAVE_NAME = "v3_step0.5_전처리(공분산저장_누적합10개_상대좌표_정육면체R2_메모리압축없음_정규화안함_법선미계산)렌더링(중앙차분법선_공분산미사용_바이섹션없음_교점은샘플점_법선계산은lighting내부).bmp"; // [harness] 파일명을 v번호_세부내용 으로 통일  
	// 추가 : 렌더 결과(MyTexture)를 24비트 BMP로 저장.
	//        BMP는 아래->위, BGR 순서로 저장한다.
	//        WIDTH*3 이 4의 배수가 아닐 경우를 대비해 패딩을 넣는다.
	//--------------------------------------------------------------------
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

		auto preStart = std::chrono::high_resolution_clock::now();
		//전처리 시작
		long long boundaryCount = 0;

		for (int z = 0; z < VOLZ; z++)
			for (int y = 0; y < VOLY; y++)
				for (int x = 0; x < VOLX; x++) {

					covVol[z][y][x].n = 0.0f;  // 기본은 비어있음 표시

					// --- 경계 판정: 6-이웃 중 자신과 다른 값이 하나라도 있으면 경계 (0쪽 1쪽 모두) ---
					if (x == 0 || y == 0 || z == 0 ||
						x == VOLX - 1 || y == VOLY - 1 || z == VOLZ - 1) continue;

					unsigned char c = vol[z][y][x];
					bool isBoundary =
						(vol[z][y][x - 1] != c) || (vol[z][y][x + 1] != c) ||
						(vol[z][y - 1][x] != c) || (vol[z][y + 1][x] != c) ||
						(vol[z - 1][y][x] != c) || (vol[z + 1][y][x] != c);
					if (!isBoundary) continue;

					boundaryCount++;

					// --- 5x5x5 창에서 값이 1인 복셀의 상대좌표를 누적 ---
					float Sxx = 0, Syy = 0, Szz = 0, Sxy = 0, Sxz = 0, Syz = 0;
					float Mx = 0, My = 0, Mz = 0;
					float n = 0;

					for (int dz = -R; dz <= R; dz++)
						for (int dy = -R; dy <= R; dy++)
							for (int dx = -R; dx <= R; dx++) {
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

					covVol[z][y][x].Mx = Mx;   covVol[z][y][x].My = My;   covVol[z][y][x].Mz = Mz;
					covVol[z][y][x].Sxx = Sxx; covVol[z][y][x].Syy = Syy; covVol[z][y][x].Szz = Szz;
					covVol[z][y][x].Sxy = Sxy; covVol[z][y][x].Sxz = Sxz; covVol[z][y][x].Syz = Syz;
					covVol[z][y][x].n = n;
				}

		auto preEnd = std::chrono::high_resolution_clock::now();
		auto preDur = std::chrono::duration_cast<std::chrono::microseconds>(preEnd - preStart);
		std::cout << "전처리(공분산) 시간: " << preDur.count() * 0.001f << " ms" << std::endl;
		std::cout << "경계 복셀 수: " << boundaryCount << std::endl;





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
		///*/*SaveBMP(SAVE_NAME);*/*///영상 저장용
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

	// ===================== [harness] 여기부터는 원본 커밋에 없는 일괄 실행용 코드 =====================
	void HarnessAlloc() { covVol = (CovData(*)[VOLY][VOLX])BigAlloc(sizeof(CovData) * VOLZ * VOLY * VOLX); }
	void HarnessFree() { BigFree(covVol); covVol = nullptr; }
	void HarnessAfterDisplay() { /*SaveBMP(SAVE_NAME);*/}   // 원본 MyDisplay 는 저장을 안 하므로 여기서 저장
} // namespace v3

// =====================================================================
//  v4   커밋 244eab0  "Jacobi_기초렌더링"
// =====================================================================
namespace v4 {

	// 시간 측정등 고성능 함수

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

	//---------- 공분산 전처리 (v3 추가) ----------
	const int R = 4;  // 이웃 반경. 5x5x5 정육면체

	struct CovData {
		float Sxx, Syy, Szz, Sxy, Sxz, Syz; // 누적합 (n으로 나누지 않음)
		float Mx, My, Mz;                   // 무게중심용 좌표 합 (상대좌표)
		float n;                            // 창 안의 1 개수
	};
	CovData(*covVol)[VOLY][VOLX] = nullptr;   // [harness] 원본: CovData covVol[VOLZ][VOLY][VOLX];

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


	//영상 저장용
	const char* SAVE_NAME = "v4_step0.5_전처리(공분산저장_상대좌표_정육면체R4_메모리압축없음_정규화안함_누적합만저장_법선미계산)렌더링(누적합보간_공분산복원_Jacobi고유분해_최소고윳값법선_부호는무게중심반대_바이섹션없음_교점은샘플점_법선계산은lighting내부).bmp"; // [harness] 파일명을 v번호_세부내용 으로 통일
	// 추가 : 렌더 결과(MyTexture)를 24비트 BMP로 저장.
	//        BMP는 아래->위, BGR 순서로 저장한다.
	//        WIDTH*3 이 4의 배수가 아닐 경우를 대비해 패딩을 넣는다.
	//--------------------------------------------------------------------
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
		//---------- v4 : 누적합 보간 -> 공분산 복원 -> Jacobi -> 법선 ----------
		int ix = int(p.x);
		int iy = int(p.y);
		int iz = int(p.z);
		float wx = p.x - ix;
		float wy = p.y - iy;
		float wz = p.z - iz;

		// 보간 결과를 담을 그릇. 0에서 시작해 8개 모서리를 더한다.
		float sxx = 0, syy = 0, szz = 0, sxy = 0, sxz = 0, syz = 0;
		float mx = 0, my = 0, mz = 0;
		float nn = 0;

		for (int dz = 0; dz < 2; dz++)
			for (int dy = 0; dy < 2; dy++)
				for (int dx = 0; dx < 2; dx++) {
					float wgt = (dx ? wx : 1.0f - wx)
						* (dy ? wy : 1.0f - wy)
						* (dz ? wz : 1.0f - wz);

					const CovData& cd = covVol[iz + dz][iy + dy][ix + dx];

					sxx += wgt * cd.Sxx;  syy += wgt * cd.Syy;  szz += wgt * cd.Szz;
					sxy += wgt * cd.Sxy;  sxz += wgt * cd.Sxz;  syz += wgt * cd.Syz;
					mx += wgt * cd.Mx;  my += wgt * cd.My;  mz += wgt * cd.Mz;
					nn += wgt * cd.n;
				}

		vec3 N(0.0f);   // 8이웃이 전부 경계가 아니면 0벡터로 남는다

		if (nn >= N_EPS) {
			double inv = 1.0 / nn;
			double C[3][3];
			C[0][0] = sxx * inv - (mx * inv) * (mx * inv);
			C[1][1] = syy * inv - (my * inv) * (my * inv);
			C[2][2] = szz * inv - (mz * inv) * (mz * inv);
			C[0][1] = C[1][0] = sxy * inv - (mx * inv) * (my * inv);
			C[0][2] = C[2][0] = sxz * inv - (mx * inv) * (mz * inv);
			C[1][2] = C[2][1] = syz * inv - (my * inv) * (mz * inv);

			double eval[3], evec[3][3];
			Jacobi3(C, eval, evec);

			// 가장 작은 고윳값의 고유벡터 = 법선
			int k = 0;
			if (eval[1] < eval[k]) k = 1;
			if (eval[2] < eval[k]) k = 2;
			double nx = evec[0][k], ny = evec[1][k], nz = evec[2][k];

			double len = sqrt(nx * nx + ny * ny + nz * nz);
			if (len > 1e-12) {
				nx /= len; ny /= len; nz /= len;
				// 부호 : 무게중심의 반대쪽이 바깥
				if (nx * (-mx) + ny * (-my) + nz * (-mz) < 0.0) {
					nx = -nx; ny = -ny; nz = -nz;
				}
				N = vec3((float)nx, (float)ny, (float)nz);
			}
		}
		//---------- v4 끝 ----------


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
						//------------------------------------------------
						// 변경 : 바이섹션 미사용. 현재 샘플점을 교점으로 삼는다.
						//------------------------------------------------
						glm::vec3 hit = p;

						glm::vec3 rgb(0.9f, 0.85f, 0.8f);
						col = lighting(hit, rgb, w);
						//---------- v4 : 공분산 보간 법선 ----------
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

					covVol[z][y][x].n = 0.0f;  // 기본은 비어있음 표시

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

					covVol[z][y][x].Mx = Mx;   covVol[z][y][x].My = My;   covVol[z][y][x].Mz = Mz;
					covVol[z][y][x].Sxx = Sxx; covVol[z][y][x].Syy = Syy; covVol[z][y][x].Szz = Szz;
					covVol[z][y][x].Sxy = Sxy; covVol[z][y][x].Sxz = Sxz; covVol[z][y][x].Syz = Syz;
					covVol[z][y][x].n = n;
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
		/*/*SaveBMP(SAVE_NAME);*////영상 저장용
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

	// ===================== [harness] 여기부터는 원본 커밋에 없는 일괄 실행용 코드 =====================
	void HarnessAlloc() { covVol = (CovData(*)[VOLY][VOLX])BigAlloc(sizeof(CovData) * VOLZ * VOLY * VOLX); }
	void HarnessFree() { BigFree(covVol); covVol = nullptr; }
	void HarnessAfterDisplay() {}   // 원본 MyDisplay 가 이미 SaveBMP(SAVE_NAME) 를 한다
} // namespace v4

// =====================================================================
//  v5   커밋 5ce98c7  "Bisect추가_R실험_구커널시도"
// =====================================================================
namespace v5 {

	// 시간 측정등 고성능 함수

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

	//---------- 공분산 전처리 (v3 추가) ----------
	const int R = 2;  // 이웃 반경. 5x5x5 정육면체

	struct CovData {
		float Sxx, Syy, Szz, Sxy, Sxz, Syz; // 누적합 (n으로 나누지 않음)
		float Mx, My, Mz;                   // 무게중심용 좌표 합 (상대좌표)
		float n;                            // 창 안의 1 개수
	};
	CovData(*covVol)[VOLY][VOLX] = nullptr;   // [harness] 원본: CovData covVol[VOLZ][VOLY][VOLX];

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


	//영상 저장용
	const char* SAVE_NAME = "v5_step0.5_전처리(공분산저장_상대좌표_정육면체R2_메모리압축없음_정규화안함_누적합만저장_법선미계산)렌더링(바이섹션10_누적합보간_공분산복원_Jacobi고유분해_최소고윳값법선_부호는무게중심반대_교점은바이섹션_법선계산은lighting내부).bmp"; // [harness] 파일명을 v번호_세부내용 으로 통일
	// 추가 : 렌더 결과(MyTexture)를 24비트 BMP로 저장.
	//        BMP는 아래->위, BGR 순서로 저장한다.
	//        WIDTH*3 이 4의 배수가 아닐 경우를 대비해 패딩을 넣는다.
	//--------------------------------------------------------------------
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
		//---------- v4 : 누적합 보간 -> 공분산 복원 -> Jacobi -> 법선 ----------
		int ix = int(p.x);
		int iy = int(p.y);
		int iz = int(p.z);
		float wx = p.x - ix;
		float wy = p.y - iy;
		float wz = p.z - iz;

		// 보간 결과를 담을 그릇. 0에서 시작해 8개 모서리를 더한다.
		float sxx = 0, syy = 0, szz = 0, sxy = 0, sxz = 0, syz = 0;
		float mx = 0, my = 0, mz = 0;
		float nn = 0;

		for (int dz = 0; dz < 2; dz++)
			for (int dy = 0; dy < 2; dy++)
				for (int dx = 0; dx < 2; dx++) {
					float wgt = (dx ? wx : 1.0f - wx)
						* (dy ? wy : 1.0f - wy)
						* (dz ? wz : 1.0f - wz);

					const CovData& cd = covVol[iz + dz][iy + dy][ix + dx];

					sxx += wgt * cd.Sxx;  syy += wgt * cd.Syy;  szz += wgt * cd.Szz;
					sxy += wgt * cd.Sxy;  sxz += wgt * cd.Sxz;  syz += wgt * cd.Syz;
					mx += wgt * cd.Mx;  my += wgt * cd.My;  mz += wgt * cd.Mz;
					nn += wgt * cd.n;
				}

		vec3 N(0.0f);   // 8이웃이 전부 경계가 아니면 0벡터로 남는다

		if (nn >= N_EPS) {
			double inv = 1.0 / nn;
			double C[3][3];
			C[0][0] = sxx * inv - (mx * inv) * (mx * inv);
			C[1][1] = syy * inv - (my * inv) * (my * inv);
			C[2][2] = szz * inv - (mz * inv) * (mz * inv);
			C[0][1] = C[1][0] = sxy * inv - (mx * inv) * (my * inv);
			C[0][2] = C[2][0] = sxz * inv - (mx * inv) * (mz * inv);
			C[1][2] = C[2][1] = syz * inv - (my * inv) * (mz * inv);

			double eval[3], evec[3][3];
			Jacobi3(C, eval, evec);

			// 가장 작은 고윳값의 고유벡터 = 법선
			int k = 0;
			if (eval[1] < eval[k]) k = 1;
			if (eval[2] < eval[k]) k = 2;
			double nx = evec[0][k], ny = evec[1][k], nz = evec[2][k];

			double len = sqrt(nx * nx + ny * ny + nz * nz);
			if (len > 1e-12) {
				nx /= len; ny /= len; nz /= len;
				// 부호 : 무게중심의 반대쪽이 바깥
				if (nx * (-mx) + ny * (-my) + nz * (-mz) < 0.0) {
					nx = -nx; ny = -ny; nz = -nz;
				}
				N = vec3((float)nx, (float)ny, (float)nz);
			}
		}
		//---------- v4 끝 ----------


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

					covVol[z][y][x].n = 0.0f;  // 기본은 비어있음 표시

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

					covVol[z][y][x].Mx = Mx;   covVol[z][y][x].My = My;   covVol[z][y][x].Mz = Mz;
					covVol[z][y][x].Sxx = Sxx; covVol[z][y][x].Syy = Syy; covVol[z][y][x].Szz = Szz;
					covVol[z][y][x].Sxy = Sxy; covVol[z][y][x].Sxz = Sxz; covVol[z][y][x].Syz = Syz;
					covVol[z][y][x].n = n;
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
		/*/*SaveBMP(SAVE_NAME);*///영상 저장용
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

	// ===================== [harness] 여기부터는 원본 커밋에 없는 일괄 실행용 코드 =====================
	void HarnessAlloc() { covVol = (CovData(*)[VOLY][VOLX])BigAlloc(sizeof(CovData) * VOLZ * VOLY * VOLX); }
	void HarnessFree() { BigFree(covVol); covVol = nullptr; }
	void HarnessAfterDisplay() {}   // 원본 MyDisplay 가 이미 SaveBMP(SAVE_NAME) 를 한다
} // namespace v5

// =====================================================================
//  v6   커밋 e9d2916  "전처리 법선 저장 + 실시간 최근접 N"
// =====================================================================
namespace v6 {

	// 시간 측정등 고성능 함수

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
	const char* SAVE_NAME = "v6_step0.5_전처리(법선저장_상대좌표_정육면체R2_메모리압축없음_전처리에서Jacobi_최소고윳값법선_부호는무게중심반대)렌더링(보간없음_최근접격자점N_바이섹션10_법선계산없음).bmp"; // [harness] 파일명을 v번호_세부내용 으로 통일
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
	NormalData(*normVol)[VOLY][VOLX] = nullptr;   // [harness] 원본: NormalData normVol[VOLZ][VOLY][VOLX];//최근접 법선용. 

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
		//---------- v6 : 보간 없음. 가장 가까운 격자점의 법선을 그대로 읽는다 ----------
		int ix = int(p.x + 0.5f);   // 반올림
		int iy = int(p.y + 0.5f);
		int iz = int(p.z + 0.5f);

		if (ix < 0) ix = 0;  if (ix > VOLX - 1) ix = VOLX - 1;
		if (iy < 0) iy = 0;  if (iy > VOLY - 1) iy = VOLY - 1;
		if (iz < 0) iz = 0;  if (iz > VOLZ - 1) iz = VOLZ - 1;

		const NormalData& nd = normVol[iz][iy][ix];
		vec3 N(nd.nx, nd.ny, nd.nz);
		//---------- v6 끝 ----------

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
		/*SaveBMP(SAVE_NAME);*///영상 저장용
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

	// ===================== [harness] 여기부터는 원본 커밋에 없는 일괄 실행용 코드 =====================
	void HarnessAlloc() { normVol = (NormalData(*)[VOLY][VOLX])BigAlloc(sizeof(NormalData) * VOLZ * VOLY * VOLX); }
	void HarnessFree() { BigFree(normVol); normVol = nullptr; }
	void HarnessAfterDisplay() {}   // 원본 MyDisplay 가 이미 SaveBMP(SAVE_NAME) 를 한다
} // namespace v6

// =====================================================================
//  v7   커밋 cb3a646  "8이웃 삼선형 N 보간"
// =====================================================================
namespace v7 {

	// 시간 측정등 고성능 함수

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
	const char* SAVE_NAME = "v7_step0.5_전처리(법선저장_상대좌표_정육면체R2_메모리압축없음_전처리에서Jacobi_최소고윳값법선_부호는무게중심반대)렌더링(N삼선형보간_부호무처리_바이섹션10_법선계산없음).bmp"; // [harness] 파일명을 v번호_세부내용 으로 통일
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
	NormalData(*normVol)[VOLY][VOLX] = nullptr;   // [harness] 원본: NormalData normVol[VOLZ][VOLY][VOLX];//최근접 법선용. 

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
		/*SaveBMP(SAVE_NAME);*///영상 저장용
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

	// ===================== [harness] 여기부터는 원본 커밋에 없는 일괄 실행용 코드 =====================
	void HarnessAlloc() { normVol = (NormalData(*)[VOLY][VOLX])BigAlloc(sizeof(NormalData) * VOLZ * VOLY * VOLX); }
	void HarnessFree() { BigFree(normVol); normVol = nullptr; }
	void HarnessAfterDisplay() {}   // 원본 MyDisplay 가 이미 SaveBMP(SAVE_NAME) 를 한다
} // namespace v7

// =====================================================================
//  v8   커밋 2ed55fe  "8이웃 삼선형 보간 최근접 기분 부호 정렬"
// =====================================================================
namespace v8 {

	// 시간 측정등 고성능 함수

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
	const char* SAVE_NAME = "v8_step0.5_전처리(법선저장_상대좌표_정육면체R2_메모리압축없음_전처리에서Jacobi_최소고윳값법선_부호는무게중심반대)렌더링(N삼선형보간_중심최근접기준부호정렬_바이섹션10_법선계산없음).bmp"; // [harness] 파일명을 v번호_세부내용 으로 통일
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
	NormalData(*normVol)[VOLY][VOLX] = nullptr;   // [harness] 원본: NormalData normVol[VOLZ][VOLY][VOLX];//최근접 법선용. 

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
		//---------- v8 : 8이웃 법선 삼선형 보간. 중심(최근접) 기준 부호 정렬 ----------
		int ix = int(p.x);          // 내림. 8이웃의 기준 모서리
		int iy = int(p.y);
		int iz = int(p.z);
		float wx = p.x - ix;
		float wy = p.y - iy;
		float wz = p.z - iz;

		// 기준 법선 : 8개 중 가장 가까운 격자점의 것. 반올림과 같다.
		int rx = (wx < 0.5f) ? ix : ix + 1;
		int ry = (wy < 0.5f) ? iy : iy + 1;
		int rz = (wz < 0.5f) ? iz : iz + 1;
		const NormalData& rf = normVol[rz][ry][rx];
		vec3 Nref(rf.nx, rf.ny, rf.nz);

		vec3 N(0.0f);

		for (int dz = 0; dz < 2; dz++)
			for (int dy = 0; dy < 2; dy++)
				for (int dx = 0; dx < 2; dx++) {
					float wgt = (dx ? wx : 1.0f - wx)
						* (dy ? wy : 1.0f - wy)
						* (dz ? wz : 1.0f - wz);

					const NormalData& nd = normVol[iz + dz][iy + dy][ix + dx];
					vec3 Ni(nd.nx, nd.ny, nd.nz);

					if (dot(Ni, Nref) < 0.0f) Ni = -Ni;   // 기준과 반대편이면 뒤집는다

					N += wgt * Ni;
				}
		//---------- v8 끝 ----------

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
		/*SaveBMP(SAVE_NAME);*///영상 저장용
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

	// ===================== [harness] 여기부터는 원본 커밋에 없는 일괄 실행용 코드 =====================
	void HarnessAlloc() { normVol = (NormalData(*)[VOLY][VOLX])BigAlloc(sizeof(NormalData) * VOLZ * VOLY * VOLX); }
	void HarnessFree() { BigFree(normVol); normVol = nullptr; }
	void HarnessAfterDisplay() {}   // 원본 MyDisplay 가 이미 SaveBMP(SAVE_NAME) 를 한다
} // namespace v8

// =====================================================================
//  v9   커밋 ef54af1  "8이웃삼선형법선보간_중심최근접기준부호정렬"
// =====================================================================
namespace v9 {

	// 시간 측정등 고성능 함수

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
	const char* SAVE_NAME = "v9_step0.5_전처리(법선저장_상대좌표_정육면체R2_메모리압축없음_전처리에서Jacobi_최소고윳값법선_부호는무게중심반대)렌더링(N삼선형보간_시선기준부호정렬_바이섹션10_법선계산없음).bmp"; // [harness] 파일명을 v번호_세부내용 으로 통일
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
	NormalData(*normVol)[VOLY][VOLX] = nullptr;   // [harness] 원본: NormalData normVol[VOLZ][VOLY][VOLX];//최근접 법선용. 

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
		//---------- v9 : 8이웃 법선 삼선형 보간. 시선기준정렬 ----------
		int ix = int(p.x);          // 내림. 8이웃의 기준 모서리
		int iy = int(p.y);
		int iz = int(p.z);
		float wx = p.x - ix;
		float wy = p.y - iy;
		float wz = p.z - iz;

		// 기준 법선 : 8개 중 가장 가까운 격자점의 것. 반올림과 같다.
		vec3 Nref = -w;
		vec3 N(0.0f);

		for (int dz = 0; dz < 2; dz++)
			for (int dy = 0; dy < 2; dy++)
				for (int dx = 0; dx < 2; dx++) {
					float wgt = (dx ? wx : 1.0f - wx)
						* (dy ? wy : 1.0f - wy)
						* (dz ? wz : 1.0f - wz);

					const NormalData& nd = normVol[iz + dz][iy + dy][ix + dx];
					vec3 Ni(nd.nx, nd.ny, nd.nz);

					if (dot(Ni, Nref) < 0.0f) Ni = -Ni;   // 기준과 반대편이면 뒤집는다

					N += wgt * Ni;
				}
		//---------- v9 끝 ----------


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
		/*SaveBMP(SAVE_NAME);*///영상 저장용
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

	// ===================== [harness] 여기부터는 원본 커밋에 없는 일괄 실행용 코드 =====================
	void HarnessAlloc() { normVol = (NormalData(*)[VOLY][VOLX])BigAlloc(sizeof(NormalData) * VOLZ * VOLY * VOLX); }
	void HarnessFree() { BigFree(normVol); normVol = nullptr; }
	void HarnessAfterDisplay() {}   // 원본 MyDisplay 가 이미 SaveBMP(SAVE_NAME) 를 한다
} // namespace v9

// =====================================================================
//  v10   커밋 d4c7b63  "NNT텐서보간"
// =====================================================================
namespace v10 {

	// 시간 측정등 고성능 함수

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
	const char* SAVE_NAME = "v10_step0.5_전처리(법선저장_상대좌표_정육면체R2_메모리압축없음_전처리에서Jacobi_최소고윳값법선_부호는무게중심반대)렌더링(N삼선형보간_NNT텐서보간_픽셀마다Jacobi_최대고윳값법선_부호원리적소거_바이섹션10_법선계산은lighting내부).bmp"; // [harness] 파일명을 v번호_세부내용 으로 통일
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
	NormalData(*normVol)[VOLY][VOLX] = nullptr;   // [harness] 원본: NormalData normVol[VOLZ][VOLY][VOLX];//최근접 법선용. 

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
		//---------- v10 : 8이웃 NNᵀ 텐서 보간 -> 최대 고유벡터. 부호 없음 ----------
		int ix = int(p.x);
		int iy = int(p.y);
		int iz = int(p.z);
		float wx = p.x - ix;
		float wy = p.y - iy;
		float wz = p.z - iz;

		// 텐서 누적. 대칭이라 6개면 충분하지만, 알아보기 쉽게 3x3 그대로 쓴다.
		double T[3][3] = { {0,0,0}, {0,0,0}, {0,0,0} };

		for (int dz = 0; dz < 2; dz++)
			for (int dy = 0; dy < 2; dy++)
				for (int dx = 0; dx < 2; dx++) {
					float wgt = (dx ? wx : 1.0f - wx)
						* (dy ? wy : 1.0f - wy)
						* (dz ? wz : 1.0f - wz);

					const NormalData& nd = normVol[iz + dz][iy + dy][ix + dx];
					double a[3] = { nd.nx, nd.ny, nd.nz };

					// T += wgt * (a aᵀ).  뒤집혀도 결과가 같다.
					for (int i = 0; i < 3; i++)
						for (int j = 0; j < 3; j++)
							T[i][j] += wgt * a[i] * a[j];
				}

		vec3 N(0.0f);

		// 대각합이 0에 가까우면 8이웃이 전부 비었다는 뜻
		if (T[0][0] + T[1][1] + T[2][2] > N_EPS) {
			double eval[3], evec[3][3];
			Jacobi3(T, eval, evec);   // T는 파괴된다

			// 가장 "큰" 고윳값의 고유벡터 = 법선 (전처리와 반대이니 주의)
			int k = 0;
			if (eval[1] > eval[k]) k = 1;
			if (eval[2] > eval[k]) k = 2;

			N = vec3((float)evec[0][k], (float)evec[1][k], (float)evec[2][k]);
		}
		//---------- v10 끝 ----------


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
		/*SaveBMP(SAVE_NAME);*///영상 저장용
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

	// ===================== [harness] 여기부터는 원본 커밋에 없는 일괄 실행용 코드 =====================
	void HarnessAlloc() { normVol = (NormalData(*)[VOLY][VOLX])BigAlloc(sizeof(NormalData) * VOLZ * VOLY * VOLX); }
	void HarnessFree() { BigFree(normVol); normVol = nullptr; }
	void HarnessAfterDisplay() {}   // 원본 MyDisplay 가 이미 SaveBMP(SAVE_NAME) 를 한다
} // namespace v10

// =====================================================================
//  v11   커밋 6486ce6  "부호장 기준 정렬"
// =====================================================================
namespace v11 {

	// 시간 측정등 고성능 함수

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
	const char* SAVE_NAME = "v11_step0.5_전처리(법선저장_상대좌표_정육면체R2_메모리압축없음_전처리에서Jacobi_최소고윳값법선_부호는무게중심반대)렌더링(N삼선형보간_부호장기울기기준정렬_바이섹션10_법선계산은lighting내부).bmp"; // [harness] 파일명을 v번호_세부내용 으로 통일
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
	NormalData(*normVol)[VOLY][VOLX] = nullptr;   // [harness] 원본: NormalData normVol[VOLZ][VOLY][VOLX];//최근접 법선용. 

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
		//---------- v11 : 8이웃 법선 삼선형 보간. 부호장 Φ 기울기 기준 정렬 ----------
		int ix = int(p.x);
		int iy = int(p.y);
		int iz = int(p.z);
		float wx = p.x - ix;
		float wy = p.y - iy;
		float wz = p.z - iz;

		// 기준 벡터 : 밀도가 줄어드는 쪽이 바깥. 중앙차분 기울기의 반대.
		float gx = GetDensity(p + vec3(1, 0, 0)) - GetDensity(p - vec3(1, 0, 0));
		float gy = GetDensity(p + vec3(0, 1, 0)) - GetDensity(p - vec3(0, 1, 0));
		float gz = GetDensity(p + vec3(0, 0, 1)) - GetDensity(p - vec3(0, 0, 1));
		vec3 Nref(-gx, -gy, -gz);

		vec3 N(0.0f);

		for (int dz = 0; dz < 2; dz++)
			for (int dy = 0; dy < 2; dy++)
				for (int dx = 0; dx < 2; dx++) {
					float wgt = (dx ? wx : 1.0f - wx)
						* (dy ? wy : 1.0f - wy)
						* (dz ? wz : 1.0f - wz);

					const NormalData& nd = normVol[iz + dz][iy + dy][ix + dx];
					vec3 Ni(nd.nx, nd.ny, nd.nz);

					if (dot(Ni, Nref) < 0.0f) Ni = -Ni;

					N += wgt * Ni;
				}
		//---------- v11 끝 ----------

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
		/*SaveBMP(SAVE_NAME);*///영상 저장용
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

	// ===================== [harness] 여기부터는 원본 커밋에 없는 일괄 실행용 코드 =====================
	void HarnessAlloc() { normVol = (NormalData(*)[VOLY][VOLX])BigAlloc(sizeof(NormalData) * VOLZ * VOLY * VOLX); }
	void HarnessFree() { BigFree(normVol); normVol = nullptr; }
	void HarnessAfterDisplay() {}   // 원본 MyDisplay 가 이미 SaveBMP(SAVE_NAME) 를 한다
} // namespace v11

// =====================================================================
//  v12   커밋 7f45039  "v12:실시간"
// =====================================================================
namespace v12 {

	// 시간 측정등 고성능 함수

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
	const char* SAVE_NAME = "v12_step0.5_전처리(저장없음_이진화만)렌더링(실시간_교점중심정육면체R2_실수상대좌표_Jacobi_최소고윳값법선_부호는무게중심반대_보간없음_바이섹션10회_법선계산은lighting내부).bmp"; // [harness] 파일명을 v번호_세부내용 으로 통일
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
	//NormalData normVol[VOLZ][VOLY][VOLX];//최근접 법선용.  v12는 실시간이라 저장 안해용. 

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
		//---------- v12 : 실시간. 교점 p 중심 정육면체 창에서 바로 PCA ----------
	// 창 범위 : 각 축으로 p-R ~ p+R 안에 드는 정수 복셀
		int x0 = (int)ceil(p.x - R);
		int x1 = (int)floor(p.x + R);
		int y0 = (int)ceil(p.y - R);
		int y1 = (int)floor(p.y + R);
		int z0 = (int)ceil(p.z - R);
		int z1 = (int)floor(p.z + R);

		double Sxx = 0, Syy = 0, Szz = 0, Sxy = 0, Sxz = 0, Syz = 0;
		double Mx = 0, My = 0, Mz = 0;
		double n = 0;

		for (int z = z0; z <= z1; z++)
			for (int y = y0; y <= y1; y++)
				for (int x = x0; x <= x1; x++) {
					if (x < 0) continue;
					if (y < 0) continue;
					if (z < 0) continue;
					if (x >= VOLX) continue;
					if (y >= VOLY) continue;
					if (z >= VOLZ) continue;
					if (vol[z][y][x] == 0) continue;

					// 상대좌표 : 원점이 p. 실수가 된다
					double fx = x - p.x;
					double fy = y - p.y;
					double fz = z - p.z;
					Mx += fx;  My += fy;  Mz += fz;
					Sxx += fx * fx;  Syy += fy * fy;  Szz += fz * fz;
					Sxy += fx * fy;  Sxz += fx * fz;  Syz += fy * fz;
					n += 1.0;
				}

		vec3 N(0.0f);   // 실패하면 0벡터 그대로

		if (n >= N_EPS) {
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
			if (len >= 1e-12) {
				nx /= len; ny /= len; nz /= len;

				// 부호 : 무게중심의 반대쪽이 바깥
				if (nx * (-Mx) + ny * (-My) + nz * (-Mz) < 0.0) {
					nx = -nx; ny = -ny; nz = -nz;
				}
				N = vec3((float)nx, (float)ny, (float)nz);
			}
		}
		//---------- v12 끝 ----------

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

		//for (int z = 0; z < VOLZ; z++)
		//	for (int y = 0; y < VOLY; y++)
		//		for (int x = 0; x < VOLX; x++) {

		//			//covVol[z][y][x].n = 0.0f;  // 기본은 비어있음 표시

		//			// --- 경계 판정: 6-이웃 중 자신과 다른 값이 하나라도 있으면 경계 (0쪽 1쪽 모두) ---
		//			if (x == 0 || y == 0 || z == 0 ||
		//				x == VOLX - 1 || y == VOLY - 1 || z == VOLZ - 1) continue;

		//			unsigned char c = vol[z][y][x];
		//			bool isBoundary =
		//				(vol[z][y][x - 1] != c) || (vol[z][y][x + 1] != c) ||
		//				(vol[z][y - 1][x] != c) || (vol[z][y + 1][x] != c) ||
		//				(vol[z - 1][y][x] != c) || (vol[z + 1][y][x] != c);
		//			if (!isBoundary) continue;

		//			// --- 5x5x5 창에서 값이 1인 복셀의 상대좌표!!를 누적 ---
		//			float Sxx = 0, Syy = 0, Szz = 0, Sxy = 0, Sxz = 0, Syz = 0;
		//			float Mx = 0, My = 0, Mz = 0;
		//			float n = 0;

		//			for (int dz = -R; dz <= R; dz++)
		//				for (int dy = -R; dy <= R; dy++)
		//					for (int dx = -R; dx <= R; dx++) {
		//						//if (dx * dx + dy * dy + dz * dz > R * R) continue;
		//						int nx = x + dx, ny = y + dy, nz = z + dz;
		//						if (nx < 0 || ny < 0 || nz < 0 ||
		//							nx >= VOLX || ny >= VOLY || nz >= VOLZ) continue;
		//						if (vol[nz][ny][nx] == 0) continue;

		//						float fx = (float)dx, fy = (float)dy, fz = (float)dz;
		//						Mx += fx;  My += fy;  Mz += fz;
		//						Sxx += fx * fx;  Syy += fy * fy;  Szz += fz * fz;
		//						Sxy += fx * fy;  Sxz += fx * fz;  Syz += fy * fz;
		//						n += 1.0f;
		//					}

		//			/*covVol[z][y][x].Mx = Mx;   covVol[z][y][x].My = My;   covVol[z][y][x].Mz = Mz;
		//			covVol[z][y][x].Sxx = Sxx; covVol[z][y][x].Syy = Syy; covVol[z][y][x].Szz = Szz;
		//			covVol[z][y][x].Sxy = Sxy; covVol[z][y][x].Sxz = Sxz; covVol[z][y][x].Syz = Syz;
		//			covVol[z][y][x].n = n;*/
		//			//---------- v6 : 여기서 바로 공분산 복원 -> Jacobi -> 법선 ----------
		//			if (n < N_EPS) continue;   // 창이 비었으면 법선 없음 (0,0,0) 유지

		//			double inv = 1.0 / n;
		//			double C[3][3];
		//			C[0][0] = Sxx * inv - (Mx * inv) * (Mx * inv);
		//			C[1][1] = Syy * inv - (My * inv) * (My * inv);
		//			C[2][2] = Szz * inv - (Mz * inv) * (Mz * inv);
		//			C[0][1] = C[1][0] = Sxy * inv - (Mx * inv) * (My * inv);
		//			C[0][2] = C[2][0] = Sxz * inv - (Mx * inv) * (Mz * inv);
		//			C[1][2] = C[2][1] = Syz * inv - (My * inv) * (Mz * inv);

		//			double eval[3], evec[3][3];
		//			Jacobi3(C, eval, evec);

		//			// 가장 작은 고윳값의 고유벡터 = 법선
		//			int k = 0;
		//			if (eval[1] < eval[k]) k = 1;
		//			if (eval[2] < eval[k]) k = 2;
		//			double nx = evec[0][k], ny = evec[1][k], nz = evec[2][k];

		//			double len = sqrt(nx * nx + ny * ny + nz * nz);
		//			if (len < 1e-12) continue;   // 실패. (0,0,0) 유지
		//			nx /= len; ny /= len; nz /= len;

		//			// 부호 : 무게중심의 반대쪽이 바깥
		//			if (nx * (-Mx) + ny * (-My) + nz * (-Mz) < 0.0) {
		//				nx = -nx; ny = -ny; nz = -nz;
		//			}

		//			normVol[z][y][x].nx = (float)nx;
		//			normVol[z][y][x].ny = (float)ny;
		//			normVol[z][y][x].nz = (float)nz;
		//		}
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
		//SaveBMP(SAVE_NAME);//영상 저장용
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

	// ===================== [harness] 여기부터는 원본 커밋에 없는 일괄 실행용 코드 =====================
	void HarnessAlloc() {}   // 이 버전은 거대 배열이 없다
	void HarnessFree() {}
	void HarnessAfterDisplay() {}   // 원본 MyDisplay 가 이미 SaveBMP(SAVE_NAME) 를 한다
} // namespace v12

// =====================================================================
//  v13   커밋 078f95b  "v13 공분산을 전처리로 저장"
// =====================================================================
namespace v13 {

	// 시간 측정등 고성능 함수

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
	const char* SAVE_NAME = "v13_step0.5_전처리(공분산6개저장_상대좌표_정육면체R2_n으로나눔_무게중심버림_법선미계산)렌더링(공분산6개삼선형보간_픽셀마다Jacobi_최소고윳값법선_부호무처리_조명fabs의존_바이섹션10회_법선계산은lighting내부).bmp"; // [harness] 파일명을 v번호_세부내용 으로 통일
	//---------- 공분산 전처리 (v3 추가) ----------
	const int R = 2;  // 이웃 반경. 5x5x5 정육면체
	//
	//struct CovData {
	//	float Sxx, Syy, Szz, Sxy, Sxz, Syz; // 누적합 (n으로 나누지 않음)
	//	float Mx, My, Mz;                   // 무게중심용 좌표 합 (상대좌표)
	//	float n;                            // 창 안의 1 개수
	//};
	//CovData covVol[VOLZ][VOLY][VOLX];
	//struct NormalData {
	//	float nx, ny, nz;   // 단위 법선. 경계가 아니거나 실패하면 (0,0,0)
	//};
	//NormalData normVol[VOLZ][VOLY][VOLX];//최근접 법선용.  v12는 실시간이라 저장 안해용. 

	// v13 : 공분산 6개만 저장. M과 n은 버린다 (24바이트)
	struct CovData6 {
		float Cxx, Cyy, Czz, Cxy, Cxz, Cyz;
	};
	CovData6(*covVol)[VOLY][VOLX] = nullptr;   // [harness] 원본: CovData6 covVol[VOLZ][VOLY][VOLX];

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
		//---------- v13 : 공분산 6개 삼선형 보간 -> 픽셀마다 Jacobi ----------
		int ix = int(p.x);
		int iy = int(p.y);
		int iz = int(p.z);
		float wx = p.x - ix;
		float wy = p.y - iy;
		float wz = p.z - iz;

		double Cxx = 0, Cyy = 0, Czz = 0, Cxy = 0, Cxz = 0, Cyz = 0;

		for (int dz = 0; dz < 2; dz++)
			for (int dy = 0; dy < 2; dy++)
				for (int dx = 0; dx < 2; dx++) {
					double wgt = (dx ? wx : 1.0f - wx)
						* (dy ? wy : 1.0f - wy)
						* (dz ? wz : 1.0f - wz);

					const CovData6& cd = covVol[iz + dz][iy + dy][ix + dx];
					Cxx += wgt * cd.Cxx;
					Cyy += wgt * cd.Cyy;
					Czz += wgt * cd.Czz;
					Cxy += wgt * cd.Cxy;
					Cxz += wgt * cd.Cxz;
					Cyz += wgt * cd.Cyz;
				}

		double C[3][3];
		C[0][0] = Cxx;  C[1][1] = Cyy;  C[2][2] = Czz;
		C[0][1] = C[1][0] = Cxy;
		C[0][2] = C[2][0] = Cxz;
		C[1][2] = C[2][1] = Cyz;

		double eval[3], evec[3][3];
		Jacobi3(C, eval, evec);

		// 가장 작은 고윳값의 고유벡터 = 법선. 부호는 정하지 않는다
		int k = 0;
		if (eval[1] < eval[k]) k = 1;
		if (eval[2] < eval[k]) k = 2;

		vec3 N((float)evec[0][k], (float)evec[1][k], (float)evec[2][k]);
		//---------- v13 끝 ----------

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

					//---------- v13 : 공분산만 복원해서 저장. Jacobi 없음 ----------
					if (n < N_EPS) continue;   // 창이 비었으면 (0,...,0) 유지

					float inv = 1.0f / n;
					covVol[z][y][x].Cxx = Sxx * inv - (Mx * inv) * (Mx * inv);
					covVol[z][y][x].Cyy = Syy * inv - (My * inv) * (My * inv);
					covVol[z][y][x].Czz = Szz * inv - (Mz * inv) * (Mz * inv);
					covVol[z][y][x].Cxy = Sxy * inv - (Mx * inv) * (My * inv);
					covVol[z][y][x].Cxz = Sxz * inv - (Mx * inv) * (Mz * inv);
					covVol[z][y][x].Cyz = Syz * inv - (My * inv) * (Mz * inv);
					//---------- v13 끝 ----------
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
		/*SaveBMP(SAVE_NAME);*///영상 저장용
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

	// ===================== [harness] 여기부터는 원본 커밋에 없는 일괄 실행용 코드 =====================
	void HarnessAlloc() { covVol = (CovData6(*)[VOLY][VOLX])BigAlloc(sizeof(CovData6) * VOLZ * VOLY * VOLX); }
	void HarnessFree() { BigFree(covVol); covVol = nullptr; }
	void HarnessAfterDisplay() {}   // 원본 MyDisplay 가 이미 SaveBMP(SAVE_NAME) 를 한다
} // namespace v13

// =====================================================================
//  v14   커밋 1d679df  "v14 텐서 저장 at 전처리"
// =====================================================================
namespace v14 {

	// 시간 측정등 고성능 함수

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
	const char* SAVE_NAME = "v14_step0.5_전처리(NNT텐서6개저장_상대좌표_정육면체R2_전처리에서Jacobi_최소고윳값법선_부호없음)렌더링(텐서6개삼선형보간_픽셀마다Jacobi_최대고윳값법선_부호원리적소거_바이섹션10회_법선계산은lighting내부).bmp"; // [harness] 파일명을 v번호_세부내용 으로 통일//---------- 공분산 전처리 (v3 추가) ----------
	const int R = 2;  // 이웃 반경. 5x5x5 정육면체
	//
	//struct CovData {
	//	float Sxx, Syy, Szz, Sxy, Sxz, Syz; // 누적합 (n으로 나누지 않음)
	//	float Mx, My, Mz;                   // 무게중심용 좌표 합 (상대좌표)
	//	float n;                            // 창 안의 1 개수
	//};
	//CovData covVol[VOLZ][VOLY][VOLX];
	//struct NormalData {
	//	float nx, ny, nz;   // 단위 법선. 경계가 아니거나 실패하면 (0,0,0)
	//};
	//NormalData normVol[VOLZ][VOLY][VOLX];//최근접 법선용.  v12는 실시간이라 저장 안해용. 

	// v13 : 공분산 6개만 저장. M과 n은 버린다 (24바이트)
	//struct CovData6 {
	//	float Cxx, Cyy, Czz, Cxy, Cxz, Cyz;
	//};
	//CovData6 covVol[VOLZ][VOLY][VOLX];
	// v14 : NNT 텐서 6개 저장 (대칭이라 6개, 24바이트)
	struct TensorData {
		float Txx, Tyy, Tzz, Txy, Txz, Tyz;
	};
	TensorData(*tenVol)[VOLY][VOLX] = nullptr;   // [harness] 원본: TensorData tenVol[VOLZ][VOLY][VOLX];

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
		//---------- v14 : NNT 텐서 6개 삼선형 보간 -> 픽셀마다 Jacobi ----------
		int ix = int(p.x);
		int iy = int(p.y);
		int iz = int(p.z);
		float wx = p.x - ix;
		float wy = p.y - iy;
		float wz = p.z - iz;

		double Txx = 0, Tyy = 0, Tzz = 0, Txy = 0, Txz = 0, Tyz = 0;

		for (int dz = 0; dz < 2; dz++)
			for (int dy = 0; dy < 2; dy++)
				for (int dx = 0; dx < 2; dx++) {
					double wgt = (dx ? wx : 1.0f - wx)
						* (dy ? wy : 1.0f - wy)
						* (dz ? wz : 1.0f - wz);

					const TensorData& td = tenVol[iz + dz][iy + dy][ix + dx];
					Txx += wgt * td.Txx;
					Tyy += wgt * td.Tyy;
					Tzz += wgt * td.Tzz;
					Txy += wgt * td.Txy;
					Txz += wgt * td.Txz;
					Tyz += wgt * td.Tyz;
				}

		double T[3][3];
		T[0][0] = Txx;  T[1][1] = Tyy;  T[2][2] = Tzz;
		T[0][1] = T[1][0] = Txy;
		T[0][2] = T[2][0] = Txz;
		T[1][2] = T[2][1] = Tyz;

		double eval[3], evec[3][3];
		Jacobi3(T, eval, evec);

		// 가장 "큰" 고윳값의 고유벡터 = 법선. 부호는 원리적으로 없다
		int k = 0;
		if (eval[1] > eval[k]) k = 1;
		if (eval[2] > eval[k]) k = 2;

		vec3 N((float)evec[0][k], (float)evec[1][k], (float)evec[2][k]);
		//---------- v14 끝 ----------

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

					//---------- v14 : 법선을 구한 뒤 NNT로 만들어 저장 ----------
					if (n < N_EPS) continue;

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

					int k = 0;
					if (eval[1] < eval[k]) k = 1;
					if (eval[2] < eval[k]) k = 2;
					double nx = evec[0][k], ny = evec[1][k], nz = evec[2][k];

					double len = sqrt(nx * nx + ny * ny + nz * nz);
					if (len < 1e-12) continue;   // 실패. (0,...,0) 유지
					nx /= len; ny /= len; nz /= len;

					// 부호 판정 없음. NNT는 N을 뒤집어도 같다
					tenVol[z][y][x].Txx = (float)(nx * nx);
					tenVol[z][y][x].Tyy = (float)(ny * ny);
					tenVol[z][y][x].Tzz = (float)(nz * nz);
					tenVol[z][y][x].Txy = (float)(nx * ny);
					tenVol[z][y][x].Txz = (float)(nx * nz);
					tenVol[z][y][x].Tyz = (float)(ny * nz);
					//---------- v14 끝 ----------
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
		///*SaveBMP(SAVE_NAME);*///영상 저장용
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

	// ===================== [harness] 여기부터는 원본 커밋에 없는 일괄 실행용 코드 =====================
	void HarnessAlloc() { tenVol = (TensorData(*)[VOLY][VOLX])BigAlloc(sizeof(TensorData) * VOLZ * VOLY * VOLX); }
	void HarnessFree() { BigFree(tenVol); tenVol = nullptr; }
	void HarnessAfterDisplay() {}   // 원본 MyDisplay 가 이미 SaveBMP(SAVE_NAME) 를 한다
} // namespace v14

// =====================================================================
//  [harness] 일괄 실행
// =====================================================================
typedef void (*HarnessFn)();

static void RunVersion(const char* label, HarnessFn alloc, HarnessFn init, HarnessFn display, HarnessFn afterDisplay, HarnessFn release, bool hasPreTimer)
{
	std::cout << label << std::endl;

	alloc();
	init();                 // 전처리. 원본 코드가 "전처리(공분산) 시간"을 직접 찍는다
	if (!hasPreTimer)
		std::cout << "전처리(공분산) 시간: 없음 (이 버전에는 공분산 전처리가 없다)" << std::endl;
	display();              // 1프레임. 원본 코드가 "실행 시간"과 "saved : ..." 를 직접 찍는다
	afterDisplay();         // 원본 MyDisplay 가 저장을 안 하는 버전(v1, v3)만 여기서 저장
	release();

	std::cout << "\n\n" << std::flush;   // 줄바꿈 두 번
}

int main(int argc, char** argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutCreateWindow("OpenGL Drawing Example");

	RunVersion("v1", v1::HarnessAlloc, v1::MyInit, v1::MyDisplay, v1::HarnessAfterDisplay, v1::HarnessFree, false);
	RunVersion("v2", v2::HarnessAlloc, v2::MyInit, v2::MyDisplay, v2::HarnessAfterDisplay, v2::HarnessFree, false);
	RunVersion("v3", v3::HarnessAlloc, v3::MyInit, v3::MyDisplay, v3::HarnessAfterDisplay, v3::HarnessFree, true);
	RunVersion("v4", v4::HarnessAlloc, v4::MyInit, v4::MyDisplay, v4::HarnessAfterDisplay, v4::HarnessFree, true);
	RunVersion("v5", v5::HarnessAlloc, v5::MyInit, v5::MyDisplay, v5::HarnessAfterDisplay, v5::HarnessFree, true);
	RunVersion("v6", v6::HarnessAlloc, v6::MyInit, v6::MyDisplay, v6::HarnessAfterDisplay, v6::HarnessFree, true);
	RunVersion("v7", v7::HarnessAlloc, v7::MyInit, v7::MyDisplay, v7::HarnessAfterDisplay, v7::HarnessFree, true);
	RunVersion("v8", v8::HarnessAlloc, v8::MyInit, v8::MyDisplay, v8::HarnessAfterDisplay, v8::HarnessFree, true);
	RunVersion("v9", v9::HarnessAlloc, v9::MyInit, v9::MyDisplay, v9::HarnessAfterDisplay, v9::HarnessFree, true);
	RunVersion("v10", v10::HarnessAlloc, v10::MyInit, v10::MyDisplay, v10::HarnessAfterDisplay, v10::HarnessFree, true);
	RunVersion("v11", v11::HarnessAlloc, v11::MyInit, v11::MyDisplay, v11::HarnessAfterDisplay, v11::HarnessFree, true);
	RunVersion("v12", v12::HarnessAlloc, v12::MyInit, v12::MyDisplay, v12::HarnessAfterDisplay, v12::HarnessFree, true);
	RunVersion("v13", v13::HarnessAlloc, v13::MyInit, v13::MyDisplay, v13::HarnessAfterDisplay, v13::HarnessFree, true);
	RunVersion("v14", v14::HarnessAlloc, v14::MyInit, v14::MyDisplay, v14::HarnessAfterDisplay, v14::HarnessFree, true);

	std::cout << "끝. Enter 를 누르면 종료합니다." << std::endl;
	std::cin.get();
	return 0;
}
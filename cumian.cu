// v0 : GPU 쪽 (커널 함수 + 커널을 부르는 얇은 창구 함수)
// cpp 쪽 MyInit / Render 가 아래 창구 함수(GpuStep1, GpuStep2, GpuRender)를 부른다.
#include <cuda_runtime.h>
#include <math.h>

// ---- cpp 와 반드시 같은 값이어야 하는 상수 (cpp 를 바꾸면 여기도 바꿀 것) ----
#define WIDTH   512
#define HEIGHT  512
#define VOLX 256
#define VOLY 256
#define VOLZ 225
const int BSIZE = 8;
const int BSHIFT = 3;
const int BZ_COUNT = VOLZ / BSIZE + (VOLZ % BSIZE != 0);
const int BY_COUNT = VOLY / BSIZE + (VOLY % BSIZE != 0);
const int BX_COUNT = VOLX / BSIZE + (VOLX % BSIZE != 0);

//----------------------------- GPU 쪽 함수 (예외적으로 MyInit 밖) -----------------------------
// GetDensity 와 같은 삼선형 보간
__device__ float dDensity(const unsigned char* V, float px, float py, float pz) {
	int ix = (int)px, iy = (int)py, iz = (int)pz;
	float wx = px - ix, wy = py - iy, wz = pz - iz;
	int i000 = (iz * VOLY + iy) * VOLX + ix;
	int DX = 1, DY = VOLX, DZ = VOLX * VOLY;
	float den = V[i000] * (1 - wx) * (1 - wy) * (1 - wz)
		+ V[i000 + DX] * (wx) * (1 - wy) * (1 - wz)
		+ V[i000 + DY] * (1 - wx) * (wy) * (1 - wz)
		+ V[i000 + DX + DY] * (wx) * (wy) * (1 - wz)
		+ V[i000 + DZ] * (1 - wx) * (1 - wy) * (wz)
		+V[i000 + DZ + DX] * (wx) * (1 - wy) * (wz)
		+V[i000 + DZ + DY] * (1 - wx) * (wy) * (wz)
		+V[i000 + DZ + DX + DY] * (wx) * (wy) * (wz);
	return den;
}

// isOutside 와 같음 (조건을 if 로 나눔)
__device__ bool dIsOutside(float px, float py, float pz) {
	if (px >= VOLX - 1) return true;
	if (px < 0) return true;
	if (py >= VOLY - 1) return true;
	if (py < 0) return true;
	if (pz >= VOLZ - 1) return true;
	if (pz < 0) return true;
	return false;
}

// Phi 와 같음 : 안쪽 양수, 바깥 음수
__device__ float dPhi(const unsigned char* V, float px, float py, float pz) {
	// ISO_LEVEL(0.5) 대신 숫자로 적음 (cpp 쪽 상수라 여기서 안 보임)
	if (dIsOutside(px, py, pz)) return 0.0f - 0.5f;
	return dDensity(V, px, py, pz) - 0.5f;
}

// Jacobi 고유분해 (이전 1단계 코드 그대로). 끝나면 A 대각 = 고윳값, V 열 = 고유벡터
__device__ void dJacobi(float A[3][3], float V[3][3]) {
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			V[i][j] = (i == j) ? 1.0f : 0.0f;
	for (int sweep = 0; sweep < 50; sweep++) {
		float off = A[0][1] * A[0][1] + A[0][2] * A[0][2] + A[1][2] * A[1][2];
		if (off < 1e-12f) break;
		for (int p = 0; p < 2; p++)
			for (int q = p + 1; q < 3; q++) {
				if (fabsf(A[p][q]) < 1e-20f) continue;
				float th = (A[q][q] - A[p][p]) / (2.0f * A[p][q]);
				float tt = ((th >= 0) ? 1.0f : -1.0f) / (fabsf(th) + sqrtf(th * th + 1.0f));
				float cs = 1.0f / sqrtf(tt * tt + 1.0f), sn = tt * cs;
				for (int k = 0; k < 3; k++) {          // 열 회전
					float akp = A[k][p], akq = A[k][q];
					A[k][p] = cs * akp - sn * akq;  A[k][q] = sn * akp + cs * akq;
				}
				for (int k = 0; k < 3; k++) {          // 행 회전
					float apk = A[p][k], aqk = A[q][k];
					A[p][k] = cs * apk - sn * aqk;  A[q][k] = sn * apk + cs * aqk;
				}
				for (int k = 0; k < 3; k++) {          // 고유벡터 누적
					float vkp = V[k][p], vkq = V[k][q];
					V[k][p] = cs * vkp - sn * vkq;  V[k][q] = sn * vkp + cs * vkq;
				}
			}
	}
}

// 6성분 c -> 3x3, Jacobi, 큰 순서 번호 o[0] >= o[1] >= o[2]
__device__ void dEigSorted(const float* c, float lam[3], float V[3][3], int o[3]) {
	float M[3][3];
	M[0][0] = c[0]; M[1][1] = c[1]; M[2][2] = c[2];
	M[0][1] = M[1][0] = c[3];
	M[0][2] = M[2][0] = c[4];
	M[1][2] = M[2][1] = c[5];
	dJacobi(M, V);
	float l[3] = { M[0][0], M[1][1], M[2][2] };
	o[0] = 0; o[1] = 1; o[2] = 2;
	int tmp;
	if (l[o[0]] < l[o[1]]) { tmp = o[0]; o[0] = o[1]; o[1] = tmp; }
	if (l[o[1]] < l[o[2]]) { tmp = o[1]; o[1] = o[2]; o[2] = tmp; }
	if (l[o[0]] < l[o[1]]) { tmp = o[0]; o[0] = o[1]; o[1] = tmp; }
	lam[0] = l[o[0]]; lam[1] = l[o[1]]; lam[2] = l[o[2]];
}

//==================== 1단계 커널 : 격자점 하나 = 스레드 하나, z 한 장씩 ====================
__global__ void K_Step1(const unsigned char* V, float* Aout, int z, int R, float sigma, float spacing) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= VOLX * VOLY) return;
	int x = id % VOLX, y = id / VOLX;
	int xq[3] = { x, y, z };
	int DIM[3] = { VOLX, VOLY, VOLZ };
	float inv2s2 = 1.0f / (2.0f * sigma * sigma);
	int K = int(2 * R / spacing + 0.5f);   // 창 안 가로 선 개수 - 1

	float W = 0, M[3] = { 0, 0, 0 };
	float Sxx = 0, Syy = 0, Szz = 0, Sxy = 0, Sxz = 0, Syz = 0;

	// --- 0.5 지점 수집 : 축 a 방향 선들 (이전 1단계 그대로) ---
	for (int a = 0; a < 3; a++) {
		int b = (a + 1) % 3, c = (a + 2) % 3;
		for (int ku = 0; ku <= K; ku++)
			for (int kv = 0; kv <= K; kv++) {
				float u = xq[b] - R + ku * spacing;
				float v = xq[c] - R + kv * spacing;
				if (u < 0) continue; if (u >= DIM[b] - 1) continue;
				if (v < 0) continue; if (v >= DIM[c] - 1) continue;

				for (int i = xq[a] - R; i < xq[a] + R; i++) {   // 셀 [i, i+1]
					if (i < 0) continue; if (i + 2 > DIM[a] - 1) continue;
					float q0[3], q1[3];
					q0[a] = (float)i;     q0[b] = u; q0[c] = v;
					q1[a] = (float)i + 1; q1[b] = u; q1[c] = v;
					float f0 = dDensity(V, q0[0], q0[1], q0[2]);
					float f1 = dDensity(V, q1[0], q1[1], q1[2]);
					if ((f0 - 0.5f) * (f1 - 0.5f) >= 0.0f) continue;   // 사이에 0.5 없음

					float t = (0.5f - f0) / (f1 - f0);
					float r[3];                                         // x 기준 상대좌표
					r[0] = q0[0] - x; r[1] = q0[1] - y; r[2] = q0[2] - z;
					r[a] += t;

					float wj = expf(-(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) * inv2s2);
					W += wj;
					M[0] += wj * r[0]; M[1] += wj * r[1]; M[2] += wj * r[2];
					Sxx += wj * r[0] * r[0]; Syy += wj * r[1] * r[1]; Szz += wj * r[2] * r[2];
					Sxy += wj * r[0] * r[1]; Sxz += wj * r[0] * r[2]; Syz += wj * r[1] * r[2];
				}
			}
	}
	if (W < 1e-6f) return;                                 // 띠 밖 : A = 0 그대로

	// --- 가중 무게중심 p̄ 기준 공분산 -> A 에 6성분 저장 ---
	float iW = 1.0f / W;
	float px = M[0] * iW, py = M[1] * iW, pz = M[2] * iW;
	int idx = (z * VOLY + y) * VOLX + x;
	float* A = Aout + idx * 6;
	A[0] = Sxx * iW - px * px;
	A[1] = Syy * iW - py * py;
	A[2] = Szz * iW - pz * pz;
	A[3] = Sxy * iW - px * py;
	A[4] = Sxz * iW - px * pz;
	A[5] = Syz * iW - py * pz;
}

//==================== 2단계 커널 : 팬케이크 커널 PCA ====================
__global__ void K_Step2(const unsigned char* V, const float* Ain, float* Bout, int z, float spacing,
	float sn, float stMax, float rhoFb, float rhoScale) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= VOLX * VOLY) return;
	int x = id % VOLX, y = id / VOLX;
	int idx = (z * VOLY + y) * VOLX + x;
	const float* A = Ain + idx * 6;
	float* B = Bout + idx * 6;

	// --- A 가 0 이면 띠 밖 : 건너뜀 (B = 0 그대로) ---
	int nonzero = 0;
	for (int k = 0; k < 6; k++)
		if (A[k] != 0.0f) nonzero = 1;
	if (nonzero == 0) return;

	// --- A 고유분해 -> lambda1 >= lambda2 >= lambda3, n0 = lambda3 의 고유벡터 ---
	float lam[3], Vv[3][3];
	int o[3];
	dEigSorted(A, lam, Vv, o);
	float n0[3] = { Vv[0][o[2]], Vv[1][o[2]], Vv[2][o[2]] };

	// --- rho = lambda3 / lambda_t,  lambda_t = (lambda1 + lambda2) / 2 ---
	float lt = (lam[0] + lam[1]) * 0.5f;
	int fallback = 0;
	float rho = 0.0f;
	if (lt <= 0.0f) fallback = 1;                // 점이 한 곳에 몰림 : rho 정의 불가
	else {
		rho = lam[2] / lt;
		if (rho >= rhoFb) fallback = 1;          // 평면답지 않음
	}
	if (fallback == 1) {                         // 폴백 : B = A
		for (int k = 0; k < 6; k++) B[k] = A[k];
		return;
	}

	// --- 미끄럼틀 점수 s -> sigma_t (하한 sigma_n = 공 모양) ---
	float s = 1.0f / (1.0f + (rho / rhoScale) * (rho / rhoScale));
	float st = fmaxf(stMax * s, sn);
	float invSn2 = 1.0f / (sn * sn);
	float invSt2 = 1.0f / (st * st);

	// --- 3 sigma 타원체를 딱 덮는 축정렬 상자의 반폭 (이전 v2 그대로) ---
	int xq[3] = { x, y, z };
	int DIM[3] = { VOLX, VOLY, VOLZ };
	float e[3];
	for (int k = 0; k < 3; k++)
		e[k] = 3.0f * sqrtf(st * st + (sn * sn - st * st) * n0[k] * n0[k]);

	float W = 0, M[3] = { 0, 0, 0 };
	float Sxx = 0, Syy = 0, Szz = 0, Sxy = 0, Sxz = 0, Syz = 0;
	int cnt = 0;

	// --- 0.5 지점 수집 : 1단계와 같은 방식, 범위만 상자 e 로 (이전 v2 그대로) ---
	for (int a = 0; a < 3; a++) {
		int b = (a + 1) % 3, c = (a + 2) % 3;
		int Kb = (int)floorf(e[b] / spacing);
		int Kc = (int)floorf(e[c] / spacing);
		int i0 = (int)floorf(xq[a] - e[a]);
		int i1 = (int)floorf(xq[a] + e[a]);
		for (int ku = -Kb; ku <= Kb; ku++)
			for (int kv = -Kc; kv <= Kc; kv++) {
				float u = xq[b] + ku * spacing;          // 선은 x 기준 격자 정렬
				float v = xq[c] + kv * spacing;
				if (u < 0) continue; if (u >= DIM[b] - 1) continue;
				if (v < 0) continue; if (v >= DIM[c] - 1) continue;

				for (int i = i0; i <= i1; i++) {         // 셀 [i, i+1]
					if (i < 0) continue; if (i + 2 > DIM[a] - 1) continue;
					float q0[3], q1[3];
					q0[a] = (float)i;     q0[b] = u; q0[c] = v;
					q1[a] = (float)i + 1; q1[b] = u; q1[c] = v;
					float f0 = dDensity(V, q0[0], q0[1], q0[2]);
					float f1 = dDensity(V, q1[0], q1[1], q1[2]);
					if ((f0 - 0.5f) * (f1 - 0.5f) >= 0.0f) continue;

					float t = (0.5f - f0) / (f1 - f0);
					float r[3];
					r[0] = q0[0] - x; r[1] = q0[1] - y; r[2] = q0[2] - z;
					r[a] += t;

					// 마할라노비스 거리 qm = (r.n0)^2/sn^2 + (r.r - (r.n0)^2)/st^2
					float rn = r[0] * n0[0] + r[1] * n0[1] + r[2] * n0[2];
					float r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
					float qm = rn * rn * invSn2 + (r2 - rn * rn) * invSt2;
					if (qm > 9.0f) continue;              // 3 sigma 타원체 밖 배제

					float wj = expf(-0.5f * qm);          // 팬케이크 가우시안 가중
					W += wj;
					M[0] += wj * r[0]; M[1] += wj * r[1]; M[2] += wj * r[2];
					Sxx += wj * r[0] * r[0]; Syy += wj * r[1] * r[1]; Szz += wj * r[2] * r[2];
					Sxy += wj * r[0] * r[1]; Sxz += wj * r[0] * r[2]; Syz += wj * r[1] * r[2];
					cnt++;
				}
			}
	}
	if (cnt == 0) {                              // 타원체 안 점 없음 : B = A
		for (int k = 0; k < 6; k++) B[k] = A[k];
		return;
	}

	// --- 가중 무게중심 p̄ 기준 공분산 -> B ---
	float iW = 1.0f / W;
	float px = M[0] * iW, py = M[1] * iW, pz = M[2] * iW;
	B[0] = Sxx * iW - px * px;
	B[1] = Syy * iW - py * py;
	B[2] = Szz * iW - pz * pz;
	B[3] = Sxy * iW - px * py;
	B[4] = Sxz * iW - px * pz;
	B[5] = Syz * iW - py * pz;
}

//==================== 실시간 커널 : 픽셀 하나 = 스레드 하나 ====================
__global__ void K_Render(const unsigned char* V, const unsigned char* BM, const float* Bc, unsigned char* img,
	float3 eye3, float3 u3, float3 v3, float3 w3) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= WIDTH * HEIGHT) return;
	int x = id % WIDTH, y = id / WIDTH;
	float w[3] = { w3.x, w3.y, w3.z };

	const float supersampling = 0.5f;
	float RS[3];
	RS[0] = eye3.x + u3.x * (x - WIDTH * 0.5f) * supersampling + v3.x * (y - HEIGHT * 0.5f) * supersampling;
	RS[1] = eye3.y + u3.y * (x - WIDTH * 0.5f) * supersampling + v3.y * (y - HEIGHT * 0.5f) * supersampling;
	RS[2] = eye3.z + u3.z * (x - WIDTH * 0.5f) * supersampling + v3.z * (y - HEIGHT * 0.5f) * supersampling;

	float col[3] = { 0, 0, 0 };

	// --- AABB 박스 체크 (AABB_box_check 와 같음) ---
	float t1, t2;
	t1 = -RS[0] / w[0]; t2 = ((VOLX - 1) - RS[0]) / w[0];
	float xm = fminf(t1, t2), xM = fmaxf(t1, t2);
	t1 = -RS[1] / w[1]; t2 = ((VOLY - 1) - RS[1]) / w[1];
	float ym = fminf(t1, t2), yM = fmaxf(t1, t2);
	t1 = -RS[2] / w[2]; t2 = ((VOLZ - 1) - RS[2]) / w[2];
	float zm = fminf(t1, t2), zM = fmaxf(t1, t2);
	float tm = fmaxf(fmaxf(xm, ym), zm);
	float tM = fminf(fminf(xM, yM), zM);

	if (tm < tM) {
		const float step = 0.5f;
		float tBefore = tm;
		float phiBefore = dPhi(V, RS[0] + w[0] * tm, RS[1] + w[1] * tm, RS[2] + w[2] * tm);
		for (float t = tm; t < tM; t = t + step) {
			float p[3] = { RS[0] + w[0] * t, RS[1] + w[1] * t, RS[2] + w[2] * t };
			if (dIsOutside(p[0], p[1], p[2])) continue;

			// --- 빈 블록 건너뛰기 (블록 번호를 비트 묶음 대신 세 정수로) ---
			int bx = ((int)p[0]) >> BSHIFT, by = ((int)p[1]) >> BSHIFT, bz = ((int)p[2]) >> BSHIFT;
			if (BM[(bz * BY_COUNT + by) * BX_COUNT + bx] == 0) {
				float jump = 0;
				int nbx, nby, nbz;
				do {
					jump += 1.0f;
					nbx = ((int)(p[0] + w[0] * jump)) >> BSHIFT;
					nby = ((int)(p[1] + w[1] * jump)) >> BSHIFT;
					nbz = ((int)(p[2] + w[2] * jump)) >> BSHIFT;
				} while (nbx == bx && nby == by && nbz == bz);
				t = t + (jump - step);
				tBefore = t;
				phiBefore = dPhi(V, RS[0] + w[0] * t, RS[1] + w[1] * t, RS[2] + w[2] * t);
				continue;
			}

			float phi = dPhi(V, p[0], p[1], p[2]);
			if (phiBefore * phi < 0.0f) {                       // 부호 반전 검출
				// --- 이진 바이섹션 (a = 바깥, b = 안쪽) ---
				float a[3] = { RS[0] + w[0] * tBefore, RS[1] + w[1] * tBefore, RS[2] + w[2] * tBefore };
				float b[3] = { p[0], p[1], p[2] };
				if (phiBefore > 0.0f) {                          // a 가 안쪽이면 교환
					for (int k = 0; k < 3; k++) { float tmp = a[k]; a[k] = b[k]; b[k] = tmp; }
				}
				for (int it = 0; it < 10; it++) {
					float m[3] = { (a[0] + b[0]) * 0.5f, (a[1] + b[1]) * 0.5f, (a[2] + b[2]) * 0.5f };
					if (dPhi(V, m[0], m[1], m[2]) < 0.0f) { a[0] = m[0]; a[1] = m[1]; a[2] = m[2]; }
					else { b[0] = m[0]; b[1] = m[1]; b[2] = m[2]; }
				}
				float hit[3] = { (a[0] + b[0]) * 0.5f, (a[1] + b[1]) * 0.5f, (a[2] + b[2]) * 0.5f };

				// --- 법선 : 8이웃 B 를 (확신 w x 삼선형) 가중합 ---
				int ix = (int)hit[0], iy = (int)hit[1], iz = (int)hit[2];
				float fx = hit[0] - ix, fy = hit[1] - iy, fz = hit[2] - iz;
				float S[6] = { 0, 0, 0, 0, 0, 0 };
				float wsum = 0.0f;
				for (int dz = 0; dz < 2; dz++)
					for (int dy = 0; dy < 2; dy++)
						for (int dx = 0; dx < 2; dx++) {
							int gi = ((iz + dz) * VOLY + (iy + dy)) * VOLX + (ix + dx);
							const float* c = Bc + gi * 6;
							int nonzero = 0;
							for (int k = 0; k < 6; k++)
								if (c[k] != 0.0f) nonzero = 1;
							if (nonzero == 0) continue;               // 띠 밖 격자점

							float tri = ((dx == 1) ? fx : 1 - fx) * ((dy == 1) ? fy : 1 - fy) * ((dz == 1) ? fz : 1 - fz);

							float lam[3], Vc[3][3];
							int o[3];
							dEigSorted(c, lam, Vc, o);
							float tr = c[0] + c[1] + c[2];             // lambda1 + lambda2 + lambda3
							if (tr <= 0.0f) continue;
							float wc = (lam[1] - lam[2]) / tr;
							float g = tri * wc;
							for (int k = 0; k < 6; k++) S[k] += g * (c[k] / tr);//v1-1(순수 크기 맞춤)
							wsum += g;
						}

				float N[3];
				if (wsum > 0.0f) {
					float lamS[3], VS[3][3];
					int oS[3];
					dEigSorted(S, lamS, VS, oS);
					N[0] = VS[0][oS[2]]; N[1] = VS[1][oS[2]]; N[2] = VS[2][oS[2]];   // 최소 고유벡터
				}
				else {                                           // 폴백 : 중앙차분 (진단용)
					N[0] = (dDensity(V, hit[0] + 1, hit[1], hit[2]) - dDensity(V, hit[0] - 1, hit[1], hit[2])) * 0.5f;
					N[1] = (dDensity(V, hit[0], hit[1] + 1, hit[2]) - dDensity(V, hit[0], hit[1] - 1, hit[2])) * 0.5f;
					N[2] = (dDensity(V, hit[0], hit[1], hit[2] + 1) - dDensity(V, hit[0], hit[1], hit[2] - 1)) * 0.5f;
					float len = sqrtf(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]);
					if (len > 0.0f) { N[0] /= len; N[1] /= len; N[2] /= len; }
				}

				// --- 앞뒤 : 법선이 카메라(V = -w)를 등지면 반전 ---
				float Vw[3] = { -w[0], -w[1], -w[2] };
				if (N[0] * Vw[0] + N[1] * Vw[1] + N[2] * Vw[2] < 0.0f) { N[0] = -N[0]; N[1] = -N[1]; N[2] = -N[2]; }

				// --- 조명 (lighting 과 같음) ---
				float L[3] = { -w[0], -w[1] + 0.3f, -w[2] };
				float lenL = sqrtf(L[0] * L[0] + L[1] * L[1] + L[2] * L[2]);
				L[0] /= lenL; L[1] /= lenL; L[2] /= lenL;
				float H[3] = { L[0] + Vw[0], L[1] + Vw[1], L[2] + Vw[2] };
				float lenH = sqrtf(H[0] * H[0] + H[1] * H[1] + H[2] * H[2]);
				H[0] /= lenH; H[1] /= lenH; H[2] /= lenH;
				float NL = fabsf(N[0] * L[0] + N[1] * L[1] + N[2] * L[2]);
				float NH = fabsf(N[0] * H[0] + N[1] * H[1] + N[2] * H[2]);

				float rgb[3] = { 0.9f, 0.85f, 0.8f };
				float Ks[3] = { 1.2f, 0.8f, 0.8f };
				float Ia = 0.25f, Id = 0.5f, Is = 0.9f;
				float spec = powf(NH, 30.0f);
				for (int k = 0; k < 3; k++) {
					float I = Ia * rgb[k] * 0.8f + Id * rgb[k] * NL + Is * Ks[k] * spec;
					col[k] = fminf(fmaxf(I, 0.0f), 1.0f);
				}
				break;
			}
			tBefore = t;
			phiBefore = phi;
		}
	}
	img[id * 3 + 0] = (unsigned char)(int)(col[0] * 255);
	img[id * 3 + 1] = (unsigned char)(int)(col[1] * 255);
	img[id * 3 + 2] = (unsigned char)(int)(col[2] * 255);
}

//==================== 창구 함수 : cpp 에서 부름 (여기서는 커널 실행만) ====================
void GpuStep1(const unsigned char* d_vol, float* d_A, int z, int R, float sigma, float spacing) {
	int n = VOLX * VOLY;                        // z 한 장의 격자점 수
	K_Step1 << <(n + 255) / 256, 256 >> > (d_vol, d_A, z, R, sigma, spacing);
}

void GpuStep2(const unsigned char* d_vol, const float* d_A, float* d_B, int z, float spacing,
	float sn, float stMax, float rhoFb, float rhoScale) {
	int n = VOLX * VOLY;
	K_Step2 << <(n + 255) / 256, 256 >> > (d_vol, d_A, d_B, z, spacing, sn, stMax, rhoFb, rhoScale);
}

void GpuRender(const unsigned char* d_vol, const unsigned char* d_bM, const float* d_B, unsigned char* d_img,
	const float eye[3], const float u[3], const float v[3], const float w[3]) {
	int n = WIDTH * HEIGHT;                     // 픽셀 수
	K_Render << <(n + 255) / 256, 256 >> > (d_vol, d_bM, d_B, d_img,
		make_float3(eye[0], eye[1], eye[2]), make_float3(u[0], u[1], u[2]),
		make_float3(v[0], v[1], v[2]), make_float3(w[0], w[1], w[2]));
}
// ============================================================================
// exp5_cumian.cu (미리보기용으로 확장자만 .cpp) : GPU 쪽 커널 + 창구 함수
//  exp_cumian.cu 에 (비교 방법) 커널 3개만 덧붙임. 앞부분 커널은 한 글자도 안 바꿈
//   - K_ToFloat  : 이진 볼륨(0/1) -> 실수 볼륨 g
//   - K_Blur1D   : g 를 한 축 방향으로 가우시안 평활 (세 축을 차례로 = 3차원 가우시안)
//   - K_RenderG  : K_Render 와 광선·교점·조명이 같고, 법선만 "g 의 중앙차분 그래디언트"
//                  g = 이진 볼륨 그대로 -> 중앙차분, g = 평활 볼륨 -> 가우시안 평활 후 그래디언트
// (아래는 exp_cumian 머리 주석)
// exp_cumian.cu (미리보기용으로 확장자만 .cpp) : GPU 쪽 커널 + 창구 함수
//  기준 : base 브랜치 cumian.cu (v0_베이스확정)
//  바뀐 것 (실험용 추가만, 기존 커널 K_Step1 / K_Step2 / K_Render 본문은 그대로)
//   - hVOLZ 추가 (볼륨 전체 스레드 수 계산용)
//   - K_Stats    : 띠 안 격자점 수 + 1차 rho 구간별 개수
//   - K_ToNormal : 2차 공분산 -> 최소 고유벡터 법선 3성분 (4normal 판)
//   - K_RenderN  : K_Render 복사본. 법선 부분만 "8꼭짓점 법선을 시선 쪽으로 뒤집고 삼선형 합 -> 정규화"
//   - (인공) K_RenderX : useN 0 = K_Render, 1 = K_RenderN 과 같은 계산 + 픽셀마다 최종 법선·교점을 out 에 기록
//                        (인공 데이터 정답 각도 오차용. 의료 실험 exp_main 은 이 커널을 안 씀)
// ============================================================================
#include <cuda_runtime.h>
#include <math.h>

#define WIDTH   1024
#define HEIGHT  1024
const int BSIZE = 8;
const int BSHIFT = 3;
static __constant__ int VOLX, VOLY, VOLZ, BY_COUNT, BX_COUNT;
static int hVOLX, hVOLY, hVOLZ;   // (실험) hVOLZ 추가

__device__ int dVoxel(const unsigned char* vol, int x, int y, int z) {
	return vol[(z * VOLY + y) * VOLX + x];
}

__device__ float dDensity(const unsigned char* vol, float px, float py, float pz) {
	int ix = (int)px, iy = (int)py, iz = (int)pz;
	float wx = px - ix, wy = py - iy, wz = pz - iz;
	int i000 = (iz * VOLY + iy) * VOLX + ix;
	int DX = 1, DY = VOLX, DZ = VOLX * VOLY;
	float den = vol[i000] * (1 - wx) * (1 - wy) * (1 - wz)
		+ vol[i000 + DX] * (wx) * (1 - wy) * (1 - wz)
		+ vol[i000 + DY] * (1 - wx) * (wy) * (1 - wz)
		+ vol[i000 + DX + DY] * (wx) * (wy) * (1 - wz)
		+ vol[i000 + DZ] * (1 - wx) * (1 - wy) * (wz)
		+vol[i000 + DZ + DX] * (wx) * (1 - wy) * (wz)
		+vol[i000 + DZ + DY] * (1 - wx) * (wy) * (wz)
		+vol[i000 + DZ + DX + DY] * (wx) * (wy) * (wz);
	return den;
}

__device__ bool dIsOutside(float px, float py, float pz) {
	if (px >= VOLX - 1) return true;
	if (px < 0) return true;
	if (py >= VOLY - 1) return true;
	if (py < 0) return true;
	if (pz >= VOLZ - 1) return true;
	if (pz < 0) return true;
	return false;
}

__device__ float dPhi(const unsigned char* vol, float px, float py, float pz) {
	if (dIsOutside(px, py, pz)) return 0.0f - 0.5f;
	return dDensity(vol, px, py, pz) - 0.5f;
}

__device__ void dJacobi(float A[3][3], float E[3][3]) {
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			E[i][j] = (i == j) ? 1.0f : 0.0f;
	for (int sweep = 0; sweep < 50; sweep++) {
		float off = A[0][1] * A[0][1] + A[0][2] * A[0][2] + A[1][2] * A[1][2];
		if (off < 1e-12f) break;
		for (int p = 0; p < 2; p++)
			for (int q = p + 1; q < 3; q++) {
				if (fabsf(A[p][q]) < 1e-20f) continue;
				float th = (A[q][q] - A[p][p]) / (2.0f * A[p][q]);
				float tt = ((th >= 0) ? 1.0f : -1.0f) / (fabsf(th) + sqrtf(th * th + 1.0f));
				float cs = 1.0f / sqrtf(tt * tt + 1.0f), sn = tt * cs;
				for (int k = 0; k < 3; k++) {
					float akp = A[k][p], akq = A[k][q];
					A[k][p] = cs * akp - sn * akq;  A[k][q] = sn * akp + cs * akq;
				}
				for (int k = 0; k < 3; k++) {
					float apk = A[p][k], aqk = A[q][k];
					A[p][k] = cs * apk - sn * aqk;  A[q][k] = sn * apk + cs * aqk;
				}
				for (int k = 0; k < 3; k++) {
					float ekp = E[k][p], ekq = E[k][q];
					E[k][p] = cs * ekp - sn * ekq;  E[k][q] = sn * ekp + cs * ekq;
				}
			}
	}
}

__device__ void dEigSorted(const float* c, float lam[3], float E[3][3], int o[3]) {
	float M[3][3];
	M[0][0] = c[0]; M[1][1] = c[1]; M[2][2] = c[2];
	M[0][1] = M[1][0] = c[3];
	M[0][2] = M[2][0] = c[4];
	M[1][2] = M[2][1] = c[5];
	dJacobi(M, E);
	float l[3] = { M[0][0], M[1][1], M[2][2] };
	o[0] = 0; o[1] = 1; o[2] = 2;
	int tmp;
	if (l[o[0]] < l[o[1]]) { tmp = o[0]; o[0] = o[1]; o[1] = tmp; }
	if (l[o[1]] < l[o[2]]) { tmp = o[1]; o[1] = o[2]; o[2] = tmp; }
	if (l[o[0]] < l[o[1]]) { tmp = o[0]; o[0] = o[1]; o[1] = tmp; }
	lam[0] = l[o[0]]; lam[1] = l[o[1]]; lam[2] = l[o[2]];
}

// 1단계 (기존 그대로)
__global__ void K_Step1(const unsigned char* vol, float* covA, int z, int R, float sigma) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= VOLX * VOLY) return;
	int x = id % VOLX, y = id / VOLX;
	int xq[3] = { x, y, z };
	int DIM[3] = { VOLX, VOLY, VOLZ };
	float inv2s2 = 1.0f / (2.0f * sigma * sigma);

	float W = 0, M[3] = { 0, 0, 0 };
	float Sxx = 0, Syy = 0, Szz = 0, Sxy = 0, Sxz = 0, Syz = 0;

	for (int a = 0; a < 3; a++) {
		int b = (a + 1) % 3, c = (a + 2) % 3;
		for (int pb = xq[b] - R; pb <= xq[b] + R; pb++)
			for (int pc = xq[c] - R; pc <= xq[c] + R; pc++) {
				if (pb < 0) continue; if (pb > DIM[b] - 1) continue;
				if (pc < 0) continue; if (pc > DIM[c] - 1) continue;

				for (int i = xq[a] - R; i < xq[a] + R; i++) {
					if (i < 0) continue; if (i + 1 > DIM[a] - 1) continue;
					int q0[3], q1[3];
					q0[a] = i;     q0[b] = pb; q0[c] = pc;
					q1[a] = i + 1; q1[b] = pb; q1[c] = pc;
					int v0 = dVoxel(vol, q0[0], q0[1], q0[2]);
					int v1 = dVoxel(vol, q1[0], q1[1], q1[2]);
					if (v0 + v1 != 1) continue;

					float r[3];
					r[0] = (float)(q0[0] - x); r[1] = (float)(q0[1] - y); r[2] = (float)(q0[2] - z);
					r[a] += 0.5f;

					float wj = expf(-(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) * inv2s2);
					W += wj;
					M[0] += wj * r[0]; M[1] += wj * r[1]; M[2] += wj * r[2];
					Sxx += wj * r[0] * r[0]; Syy += wj * r[1] * r[1]; Szz += wj * r[2] * r[2];
					Sxy += wj * r[0] * r[1]; Sxz += wj * r[0] * r[2]; Syz += wj * r[1] * r[2];
				}
			}
	}
	if (W < 1e-6f) return;

	float iW = 1.0f / W;
	float px = M[0] * iW, py = M[1] * iW, pz = M[2] * iW;
	float* A = covA + ((z * VOLY + y) * VOLX + x) * 6;
	A[0] = Sxx * iW - px * px;
	A[1] = Syy * iW - py * py;
	A[2] = Szz * iW - pz * pz;
	A[3] = Sxy * iW - px * py;
	A[4] = Sxz * iW - px * pz;
	A[5] = Syz * iW - py * pz;
}

// 2단계 (기존 그대로)
__global__ void K_Step2(const unsigned char* vol, const float* covA, float* covB, int z,
	float sn, float st, float sr, float rhoLo, float rhoHi) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= VOLX * VOLY) return;
	int x = id % VOLX, y = id / VOLX;
	int idx = (z * VOLY + y) * VOLX + x;
	const float* A = covA + idx * 6;
	float* B = covB + idx * 6;

	int nonzero = 0;
	for (int k = 0; k < 6; k++)
		if (A[k] != 0.0f) nonzero = 1;
	if (nonzero == 0) return;

	float lam[3], E[3][3];
	int o[3];
	dEigSorted(A, lam, E, o);
	float n0[3] = { E[0][o[2]], E[1][o[2]], E[2][o[2]] };

	float rho = (lam[1] > 0.0f) ? lam[2] / lam[1] : 1.0f;

	float t = (rho - rhoLo) / (rhoHi - rhoLo);
	if (t < 0.0f) t = 0.0f;
	if (t > 1.0f) t = 1.0f;
	float s = t * t * (3.0f - 2.0f * t);

	float snU = sn + (sr - sn) * s;
	float stU = st + (sr - st) * s;
	float invSn2 = 1.0f / (snU * snU);
	float invSt2 = 1.0f / (stU * stU);

	int xq[3] = { x, y, z };
	int DIM[3] = { VOLX, VOLY, VOLZ };
	float e[3];
	for (int k = 0; k < 3; k++)
		e[k] = 3.0f * sqrtf(stU * stU + (snU * snU - stU * stU) * n0[k] * n0[k]);

	float W = 0, M[3] = { 0, 0, 0 };
	float Sxx = 0, Syy = 0, Szz = 0, Sxy = 0, Sxz = 0, Syz = 0;

	for (int a = 0; a < 3; a++) {
		int b = (a + 1) % 3, c = (a + 2) % 3;
		int Kb = (int)floorf(e[b]);
		int Kc = (int)floorf(e[c]);
		int i0 = (int)floorf(xq[a] - e[a]);
		int i1 = (int)floorf(xq[a] + e[a]);
		for (int pb = xq[b] - Kb; pb <= xq[b] + Kb; pb++)
			for (int pc = xq[c] - Kc; pc <= xq[c] + Kc; pc++) {
				if (pb < 0) continue; if (pb > DIM[b] - 1) continue;
				if (pc < 0) continue; if (pc > DIM[c] - 1) continue;

				for (int i = i0; i <= i1; i++) {
					if (i < 0) continue; if (i + 1 > DIM[a] - 1) continue;
					int q0[3], q1[3];
					q0[a] = i;     q0[b] = pb; q0[c] = pc;
					q1[a] = i + 1; q1[b] = pb; q1[c] = pc;
					int v0 = dVoxel(vol, q0[0], q0[1], q0[2]);
					int v1 = dVoxel(vol, q1[0], q1[1], q1[2]);
					if (v0 + v1 != 1) continue;

					float r[3];
					r[0] = (float)(q0[0] - x); r[1] = (float)(q0[1] - y); r[2] = (float)(q0[2] - z);
					r[a] += 0.5f;

					float rn = r[0] * n0[0] + r[1] * n0[1] + r[2] * n0[2];
					float r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
					float qm = rn * rn * invSn2 + (r2 - rn * rn) * invSt2;
					if (qm > 9.0f) continue;

					float wj = expf(-0.5f * qm);
					W += wj;
					M[0] += wj * r[0]; M[1] += wj * r[1]; M[2] += wj * r[2];
					Sxx += wj * r[0] * r[0]; Syy += wj * r[1] * r[1]; Szz += wj * r[2] * r[2];
					Sxy += wj * r[0] * r[1]; Sxz += wj * r[0] * r[2]; Syz += wj * r[1] * r[2];
				}
			}
	}

	if (W < 1e-6f) return;

	float iW = 1.0f / W;
	float px = M[0] * iW, py = M[1] * iW, pz = M[2] * iW;
	B[0] = Sxx * iW - px * px;
	B[1] = Syy * iW - py * py;
	B[2] = Szz * iW - pz * pz;
	B[3] = Sxy * iW - px * py;
	B[4] = Sxz * iW - px * pz;
	B[5] = Syz * iW - py * pz;
}

// 실시간 (기존 그대로) : 8꼭짓점 공분산 삼선형 합 -> 최소 고유벡터
__global__ void K_Render(const unsigned char* vol, const unsigned char* bM, const float* covB, unsigned char* img,
	float3 eye3, float3 u3, float3 v3, float3 w3, float supersampling) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= WIDTH * HEIGHT) return;
	int x = id % WIDTH, y = id / WIDTH;
	float w[3] = { w3.x, w3.y, w3.z };

	float RS[3];
	RS[0] = eye3.x + u3.x * (x - WIDTH * 0.5f) * supersampling + v3.x * (y - HEIGHT * 0.5f) * supersampling;
	RS[1] = eye3.y + u3.y * (x - WIDTH * 0.5f) * supersampling + v3.y * (y - HEIGHT * 0.5f) * supersampling;
	RS[2] = eye3.z + u3.z * (x - WIDTH * 0.5f) * supersampling + v3.z * (y - HEIGHT * 0.5f) * supersampling;

	float col[3] = { 0, 0, 0 };

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
		float phiBefore = dPhi(vol, RS[0] + w[0] * tm, RS[1] + w[1] * tm, RS[2] + w[2] * tm);
		for (float t = tm; t < tM; t = t + step) {
			float p[3] = { RS[0] + w[0] * t, RS[1] + w[1] * t, RS[2] + w[2] * t };
			if (dIsOutside(p[0], p[1], p[2])) continue;

			int bx = ((int)p[0]) >> BSHIFT, by = ((int)p[1]) >> BSHIFT, bz = ((int)p[2]) >> BSHIFT;
			if (bM[(bz * BY_COUNT + by) * BX_COUNT + bx] == 0) {
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
				phiBefore = dPhi(vol, RS[0] + w[0] * t, RS[1] + w[1] * t, RS[2] + w[2] * t);
				continue;
			}

			float phi = dPhi(vol, p[0], p[1], p[2]);
			if (phiBefore * phi < 0.0f || (phi == 0.0f && phiBefore != 0.0f)) {
				float a[3] = { RS[0] + w[0] * tBefore, RS[1] + w[1] * tBefore, RS[2] + w[2] * tBefore };
				float b[3] = { p[0], p[1], p[2] };
				if (phiBefore > 0.0f) {
					for (int k = 0; k < 3; k++) { float tmp = a[k]; a[k] = b[k]; b[k] = tmp; }
				}
				for (int it = 0; it < 10; it++) {
					float m[3] = { (a[0] + b[0]) * 0.5f, (a[1] + b[1]) * 0.5f, (a[2] + b[2]) * 0.5f };
					if (dPhi(vol, m[0], m[1], m[2]) < 0.0f) { a[0] = m[0]; a[1] = m[1]; a[2] = m[2]; }
					else { b[0] = m[0]; b[1] = m[1]; b[2] = m[2]; }
				}
				float hit[3] = { (a[0] + b[0]) * 0.5f, (a[1] + b[1]) * 0.5f, (a[2] + b[2]) * 0.5f };

				int ix = (int)hit[0], iy = (int)hit[1], iz = (int)hit[2];
				float fx = hit[0] - ix, fy = hit[1] - iy, fz = hit[2] - iz;
				float S[6] = { 0, 0, 0, 0, 0, 0 };
				float wsum = 0.0f;
				for (int dz = 0; dz < 2; dz++)
					for (int dy = 0; dy < 2; dy++)
						for (int dx = 0; dx < 2; dx++) {
							const float* c = covB + (((iz + dz) * VOLY + (iy + dy)) * VOLX + (ix + dx)) * 6;
							int nonzero = 0;
							for (int k = 0; k < 6; k++)
								if (c[k] != 0.0f) nonzero = 1;
							if (nonzero == 0) continue;

							float tri = ((dx == 1) ? fx : 1 - fx) * ((dy == 1) ? fy : 1 - fy) * ((dz == 1) ? fz : 1 - fz);
							float tr = c[0] + c[1] + c[2];
							if (tr <= 0.0f) continue;

							float g = tri;
							for (int k = 0; k < 6; k++) S[k] += g * c[k];
							wsum += g;
						}

				float N[3];
				if (wsum > 0.0f) {
					float lamS[3], ES[3][3];
					int oS[3];
					dEigSorted(S, lamS, ES, oS);
					N[0] = ES[0][oS[2]]; N[1] = ES[1][oS[2]]; N[2] = ES[2][oS[2]];
				}
				else {
					N[0] = (dDensity(vol, hit[0] + 1, hit[1], hit[2]) - dDensity(vol, hit[0] - 1, hit[1], hit[2])) * 0.5f;
					N[1] = (dDensity(vol, hit[0], hit[1] + 1, hit[2]) - dDensity(vol, hit[0], hit[1] - 1, hit[2])) * 0.5f;
					N[2] = (dDensity(vol, hit[0], hit[1], hit[2] + 1) - dDensity(vol, hit[0], hit[1], hit[2] - 1)) * 0.5f;
					float len = sqrtf(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]);
					if (len > 0.0f) { N[0] /= len; N[1] /= len; N[2] /= len; }
				}

				float Vw[3] = { -w[0], -w[1], -w[2] };
				if (N[0] * Vw[0] + N[1] * Vw[1] + N[2] * Vw[2] < 0.0f) { N[0] = -N[0]; N[1] = -N[1]; N[2] = -N[2]; }

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

//==================== (실험) 추가 커널 ====================

// (실험) 띠 통계 : 스레드 하나 = 격자점 하나 (볼륨 전체)
//  cnt[0] = 1차 공분산이 0 이 아닌 격자점 (= 2단계가 도는 띠), cnt[1..3] = rho < lo / lo~hi / >= hi
//  rho 식은 K_Step2 와 같음
__global__ void K_Stats(const float* covA, unsigned long long* cnt, int nVox, float lo, float hi) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= nVox) return;
	const float* A = covA + idx * 6;
	int nonzero = 0;
	for (int k = 0; k < 6; k++)
		if (A[k] != 0.0f) nonzero = 1;
	if (nonzero == 0) return;

	float lam[3], E[3][3];
	int o[3];
	dEigSorted(A, lam, E, o);
	float rho = (lam[1] > 0.0f) ? lam[2] / lam[1] : 1.0f;

	atomicAdd(&cnt[0], 1ULL);
	if (rho < lo)      atomicAdd(&cnt[1], 1ULL);
	else if (rho < hi) atomicAdd(&cnt[2], 1ULL);
	else               atomicAdd(&cnt[3], 1ULL);
}

// (실험) 2차 공분산 -> 법선 3성분 : 스레드 하나 = 격자점 하나 (볼륨 전체)
//  렌더가 건너뛰는 꼭짓점(6성분 모두 0, 또는 대각합 <= 0)은 법선도 0 으로 둠 -> K_RenderN 이 같은 꼭짓점을 건너뜀
__global__ void K_ToNormal(const float* covB, float* nrm, int nVox) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= nVox) return;
	const float* c = covB + idx * 6;
	int nonzero = 0;
	for (int k = 0; k < 6; k++)
		if (c[k] != 0.0f) nonzero = 1;
	if (nonzero == 0) return;
	float tr = c[0] + c[1] + c[2];
	if (tr <= 0.0f) return;

	float lam[3], E[3][3];
	int o[3];
	dEigSorted(c, lam, E, o);
	float* n = nrm + idx * 3;
	n[0] = E[0][o[2]]; n[1] = E[1][o[2]]; n[2] = E[2][o[2]];
}

// (실험) 실시간, 법선 저장 판 : K_Render 와 같고 법선 부분만 다름
//  8꼭짓점 법선마다 시선 쪽으로 뒤집기(기존 규칙을 보간 전에 적용) -> 삼선형 합 -> 정규화
__global__ void K_RenderN(const unsigned char* vol, const unsigned char* bM, const float* nrm, unsigned char* img,
	float3 eye3, float3 u3, float3 v3, float3 w3, float supersampling) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= WIDTH * HEIGHT) return;
	int x = id % WIDTH, y = id / WIDTH;
	float w[3] = { w3.x, w3.y, w3.z };

	float RS[3];
	RS[0] = eye3.x + u3.x * (x - WIDTH * 0.5f) * supersampling + v3.x * (y - HEIGHT * 0.5f) * supersampling;
	RS[1] = eye3.y + u3.y * (x - WIDTH * 0.5f) * supersampling + v3.y * (y - HEIGHT * 0.5f) * supersampling;
	RS[2] = eye3.z + u3.z * (x - WIDTH * 0.5f) * supersampling + v3.z * (y - HEIGHT * 0.5f) * supersampling;

	float col[3] = { 0, 0, 0 };

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
		float phiBefore = dPhi(vol, RS[0] + w[0] * tm, RS[1] + w[1] * tm, RS[2] + w[2] * tm);
		for (float t = tm; t < tM; t = t + step) {
			float p[3] = { RS[0] + w[0] * t, RS[1] + w[1] * t, RS[2] + w[2] * t };
			if (dIsOutside(p[0], p[1], p[2])) continue;

			int bx = ((int)p[0]) >> BSHIFT, by = ((int)p[1]) >> BSHIFT, bz = ((int)p[2]) >> BSHIFT;
			if (bM[(bz * BY_COUNT + by) * BX_COUNT + bx] == 0) {
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
				phiBefore = dPhi(vol, RS[0] + w[0] * t, RS[1] + w[1] * t, RS[2] + w[2] * t);
				continue;
			}

			float phi = dPhi(vol, p[0], p[1], p[2]);
			if (phiBefore * phi < 0.0f || (phi == 0.0f && phiBefore != 0.0f)) {
				float a[3] = { RS[0] + w[0] * tBefore, RS[1] + w[1] * tBefore, RS[2] + w[2] * tBefore };
				float b[3] = { p[0], p[1], p[2] };
				if (phiBefore > 0.0f) {
					for (int k = 0; k < 3; k++) { float tmp = a[k]; a[k] = b[k]; b[k] = tmp; }
				}
				for (int it = 0; it < 10; it++) {
					float m[3] = { (a[0] + b[0]) * 0.5f, (a[1] + b[1]) * 0.5f, (a[2] + b[2]) * 0.5f };
					if (dPhi(vol, m[0], m[1], m[2]) < 0.0f) { a[0] = m[0]; a[1] = m[1]; a[2] = m[2]; }
					else { b[0] = m[0]; b[1] = m[1]; b[2] = m[2]; }
				}
				float hit[3] = { (a[0] + b[0]) * 0.5f, (a[1] + b[1]) * 0.5f, (a[2] + b[2]) * 0.5f };

				float Vw[3] = { -w[0], -w[1], -w[2] };   // (실험) 보간 전 뒤집기에 쓰려고 앞으로 옮김

				// (실험) 여기부터 K_Render 와 다름 : 8꼭짓점 법선 -> 시선 쪽 뒤집기 -> 삼선형 합
				int ix = (int)hit[0], iy = (int)hit[1], iz = (int)hit[2];
				float fx = hit[0] - ix, fy = hit[1] - iy, fz = hit[2] - iz;
				float S3[3] = { 0, 0, 0 };
				float wsum = 0.0f;
				for (int dz = 0; dz < 2; dz++)
					for (int dy = 0; dy < 2; dy++)
						for (int dx = 0; dx < 2; dx++) {
							const float* nv = nrm + (((iz + dz) * VOLY + (iy + dy)) * VOLX + (ix + dx)) * 3;
							if (nv[0] == 0.0f && nv[1] == 0.0f && nv[2] == 0.0f) continue;   // 렌더 K_Render 가 건너뛰는 꼭짓점과 같음

							float tri = ((dx == 1) ? fx : 1 - fx) * ((dy == 1) ? fy : 1 - fy) * ((dz == 1) ? fz : 1 - fz);
							float sg = (nv[0] * Vw[0] + nv[1] * Vw[1] + nv[2] * Vw[2] < 0.0f) ? -1.0f : 1.0f;
							S3[0] += tri * sg * nv[0];
							S3[1] += tri * sg * nv[1];
							S3[2] += tri * sg * nv[2];
							wsum += tri;
						}

				float N[3];
				float lenS = sqrtf(S3[0] * S3[0] + S3[1] * S3[1] + S3[2] * S3[2]);
				if (wsum > 0.0f && lenS > 0.0f) {
					N[0] = S3[0] / lenS; N[1] = S3[1] / lenS; N[2] = S3[2] / lenS;
				}
				else {   // K_Render 와 같은 중앙차분
					N[0] = (dDensity(vol, hit[0] + 1, hit[1], hit[2]) - dDensity(vol, hit[0] - 1, hit[1], hit[2])) * 0.5f;
					N[1] = (dDensity(vol, hit[0], hit[1] + 1, hit[2]) - dDensity(vol, hit[0], hit[1] - 1, hit[2])) * 0.5f;
					N[2] = (dDensity(vol, hit[0], hit[1], hit[2] + 1) - dDensity(vol, hit[0], hit[1], hit[2] - 1)) * 0.5f;
					float len = sqrtf(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]);
					if (len > 0.0f) { N[0] /= len; N[1] /= len; N[2] /= len; }
				}
				// (실험) 여기까지 다름. 아래는 K_Render 와 같음

				if (N[0] * Vw[0] + N[1] * Vw[1] + N[2] * Vw[2] < 0.0f) { N[0] = -N[0]; N[1] = -N[1]; N[2] = -N[2]; }

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

//==================== 창구 함수 ====================

void GpuSetDim(int volx, int voly, int volz, int byCount, int bxCount) {
	cudaMemcpyToSymbol(VOLX, &volx, sizeof(int));
	cudaMemcpyToSymbol(VOLY, &voly, sizeof(int));
	cudaMemcpyToSymbol(VOLZ, &volz, sizeof(int));
	cudaMemcpyToSymbol(BY_COUNT, &byCount, sizeof(int));
	cudaMemcpyToSymbol(BX_COUNT, &bxCount, sizeof(int));
	hVOLX = volx; hVOLY = voly; hVOLZ = volz;   // (실험) hVOLZ 추가
}

void GpuStep1(const unsigned char* d_vol, float* d_A, int z, int R, float sigma) {
	int n = hVOLX * hVOLY;
	K_Step1 << <(n + 255) / 256, 256 >> > (d_vol, d_A, z, R, sigma);
}

void GpuStep2(const unsigned char* d_vol, const float* d_A, float* d_B, int z,
	float sn, float st, float sr, float rhoLo, float rhoHi) {
	int n = hVOLX * hVOLY;
	K_Step2 << <(n + 255) / 256, 256 >> > (d_vol, d_A, d_B, z, sn, st, sr, rhoLo, rhoHi);
}

void GpuRender(const unsigned char* d_vol, const unsigned char* d_bM, const float* d_B, unsigned char* d_img,
	const float eye[3], const float u[3], const float v[3], const float w[3], float pixel) {
	int n = WIDTH * HEIGHT;
	K_Render << <(n + 255) / 256, 256 >> > (d_vol, d_bM, d_B, d_img,
		make_float3(eye[0], eye[1], eye[2]), make_float3(u[0], u[1], u[2]),
		make_float3(v[0], v[1], v[2]), make_float3(w[0], w[1], w[2]), pixel);
}

// (실험) 띠 통계 : 카운터를 GPU 에 잡고 세고 CPU 로 가져옴
void GpuStats(const float* d_A, unsigned long long cnt[4], float lo, float hi) {
	unsigned long long* d_cnt = 0;
	cudaMalloc(&d_cnt, 4 * sizeof(unsigned long long));
	cudaMemset(d_cnt, 0, 4 * sizeof(unsigned long long));
	int n = hVOLX * hVOLY * hVOLZ;
	K_Stats << <(n + 255) / 256, 256 >> > (d_A, d_cnt, n, lo, hi);
	cudaDeviceSynchronize();
	cudaMemcpy(cnt, d_cnt, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaFree(d_cnt);
}

// (실험) 2차 공분산 -> 법선
void GpuToNormal(const float* d_B, float* d_N) {
	int n = hVOLX * hVOLY * hVOLZ;
	K_ToNormal << <(n + 255) / 256, 256 >> > (d_B, d_N, n);
}

// (실험) 법선 저장 판 렌더
void GpuRenderN(const unsigned char* d_vol, const unsigned char* d_bM, const float* d_N, unsigned char* d_img,
	const float eye[3], const float u[3], const float v[3], const float w[3], float pixel) {
	int n = WIDTH * HEIGHT;
	K_RenderN << <(n + 255) / 256, 256 >> > (d_vol, d_bM, d_N, d_img,
		make_float3(eye[0], eye[1], eye[2]), make_float3(u[0], u[1], u[2]),
		make_float3(v[0], v[1], v[2]), make_float3(w[0], w[1], w[2]), pixel);
}

//==================== (인공) 픽셀별 법선·교점 기록 렌더 ====================
// 광선·교점·조명은 K_Render 와 같음. 법선 부분만 useN 으로 갈림
//  useN 0 : K_Render 와 같음 (8꼭짓점 공분산 삼선형 합 -> 최소 고유벡터)
//  useN 1 : K_RenderN 과 같음 (8꼭짓점 법선 시선 쪽 뒤집기 -> 삼선형 합 -> 정규화)
//  out[id*7 + 0..2] = 최종 법선 (시선 쪽으로 뒤집은 뒤), 3..5 = 교점, 6 = 1 (교점 있음) / 0 (없음)
__global__ void K_RenderX(const unsigned char* vol, const unsigned char* bM, const float* arr, int useN,
	unsigned char* img, float* out,
	float3 eye3, float3 u3, float3 v3, float3 w3, float supersampling) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= WIDTH * HEIGHT) return;
	int x = id % WIDTH, y = id / WIDTH;
	float w[3] = { w3.x, w3.y, w3.z };
	out[id * 7 + 6] = 0.0f;

	float RS[3];
	RS[0] = eye3.x + u3.x * (x - WIDTH * 0.5f) * supersampling + v3.x * (y - HEIGHT * 0.5f) * supersampling;
	RS[1] = eye3.y + u3.y * (x - WIDTH * 0.5f) * supersampling + v3.y * (y - HEIGHT * 0.5f) * supersampling;
	RS[2] = eye3.z + u3.z * (x - WIDTH * 0.5f) * supersampling + v3.z * (y - HEIGHT * 0.5f) * supersampling;

	float col[3] = { 0, 0, 0 };

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
		float phiBefore = dPhi(vol, RS[0] + w[0] * tm, RS[1] + w[1] * tm, RS[2] + w[2] * tm);
		for (float t = tm; t < tM; t = t + step) {
			float p[3] = { RS[0] + w[0] * t, RS[1] + w[1] * t, RS[2] + w[2] * t };
			if (dIsOutside(p[0], p[1], p[2])) continue;

			int bx = ((int)p[0]) >> BSHIFT, by = ((int)p[1]) >> BSHIFT, bz = ((int)p[2]) >> BSHIFT;
			if (bM[(bz * BY_COUNT + by) * BX_COUNT + bx] == 0) {
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
				phiBefore = dPhi(vol, RS[0] + w[0] * t, RS[1] + w[1] * t, RS[2] + w[2] * t);
				continue;
			}

			float phi = dPhi(vol, p[0], p[1], p[2]);
			if (phiBefore * phi < 0.0f || (phi == 0.0f && phiBefore != 0.0f)) {
				float a[3] = { RS[0] + w[0] * tBefore, RS[1] + w[1] * tBefore, RS[2] + w[2] * tBefore };
				float b[3] = { p[0], p[1], p[2] };
				if (phiBefore > 0.0f) {
					for (int k = 0; k < 3; k++) { float tmp = a[k]; a[k] = b[k]; b[k] = tmp; }
				}
				for (int it = 0; it < 10; it++) {
					float m[3] = { (a[0] + b[0]) * 0.5f, (a[1] + b[1]) * 0.5f, (a[2] + b[2]) * 0.5f };
					if (dPhi(vol, m[0], m[1], m[2]) < 0.0f) { a[0] = m[0]; a[1] = m[1]; a[2] = m[2]; }
					else { b[0] = m[0]; b[1] = m[1]; b[2] = m[2]; }
				}
				float hit[3] = { (a[0] + b[0]) * 0.5f, (a[1] + b[1]) * 0.5f, (a[2] + b[2]) * 0.5f };
				float Vw[3] = { -w[0], -w[1], -w[2] };

				int ix = (int)hit[0], iy = (int)hit[1], iz = (int)hit[2];
				float fx = hit[0] - ix, fy = hit[1] - iy, fz = hit[2] - iz;
				float N[3];
				int useFallback = 0;

				if (useN == 0) {   // K_Render 와 같음
					float S[6] = { 0, 0, 0, 0, 0, 0 };
					float wsum = 0.0f;
					for (int dz = 0; dz < 2; dz++)
						for (int dy = 0; dy < 2; dy++)
							for (int dx = 0; dx < 2; dx++) {
								const float* c = arr + (((iz + dz) * VOLY + (iy + dy)) * VOLX + (ix + dx)) * 6;
								int nonzero = 0;
								for (int k = 0; k < 6; k++)
									if (c[k] != 0.0f) nonzero = 1;
								if (nonzero == 0) continue;
								float tri = ((dx == 1) ? fx : 1 - fx) * ((dy == 1) ? fy : 1 - fy) * ((dz == 1) ? fz : 1 - fz);
								float tr = c[0] + c[1] + c[2];
								if (tr <= 0.0f) continue;
								float g = tri;
								for (int k = 0; k < 6; k++) S[k] += g * c[k];
								wsum += g;
							}
					if (wsum > 0.0f) {
						float lamS[3], ES[3][3];
						int oS[3];
						dEigSorted(S, lamS, ES, oS);
						N[0] = ES[0][oS[2]]; N[1] = ES[1][oS[2]]; N[2] = ES[2][oS[2]];
					}
					else useFallback = 1;
				}
				else {             // K_RenderN 과 같음
					float S3[3] = { 0, 0, 0 };
					float wsum = 0.0f;
					for (int dz = 0; dz < 2; dz++)
						for (int dy = 0; dy < 2; dy++)
							for (int dx = 0; dx < 2; dx++) {
								const float* nv = arr + (((iz + dz) * VOLY + (iy + dy)) * VOLX + (ix + dx)) * 3;
								if (nv[0] == 0.0f && nv[1] == 0.0f && nv[2] == 0.0f) continue;
								float tri = ((dx == 1) ? fx : 1 - fx) * ((dy == 1) ? fy : 1 - fy) * ((dz == 1) ? fz : 1 - fz);
								float sg = (nv[0] * Vw[0] + nv[1] * Vw[1] + nv[2] * Vw[2] < 0.0f) ? -1.0f : 1.0f;
								S3[0] += tri * sg * nv[0];
								S3[1] += tri * sg * nv[1];
								S3[2] += tri * sg * nv[2];
								wsum += tri;
							}
					float lenS = sqrtf(S3[0] * S3[0] + S3[1] * S3[1] + S3[2] * S3[2]);
					if (wsum > 0.0f && lenS > 0.0f) {
						N[0] = S3[0] / lenS; N[1] = S3[1] / lenS; N[2] = S3[2] / lenS;
					}
					else useFallback = 1;
				}
				if (useFallback) {   // 두 렌더와 같은 중앙차분
					N[0] = (dDensity(vol, hit[0] + 1, hit[1], hit[2]) - dDensity(vol, hit[0] - 1, hit[1], hit[2])) * 0.5f;
					N[1] = (dDensity(vol, hit[0], hit[1] + 1, hit[2]) - dDensity(vol, hit[0], hit[1] - 1, hit[2])) * 0.5f;
					N[2] = (dDensity(vol, hit[0], hit[1], hit[2] + 1) - dDensity(vol, hit[0], hit[1], hit[2] - 1)) * 0.5f;
					float len = sqrtf(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]);
					if (len > 0.0f) { N[0] /= len; N[1] /= len; N[2] /= len; }
				}

				if (N[0] * Vw[0] + N[1] * Vw[1] + N[2] * Vw[2] < 0.0f) { N[0] = -N[0]; N[1] = -N[1]; N[2] = -N[2]; }

				// 기록
				out[id * 7 + 0] = N[0]; out[id * 7 + 1] = N[1]; out[id * 7 + 2] = N[2];
				out[id * 7 + 3] = hit[0]; out[id * 7 + 4] = hit[1]; out[id * 7 + 5] = hit[2];
				out[id * 7 + 6] = 1.0f;

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

// (인공) 기록 렌더 창구
void GpuRenderX(const unsigned char* d_vol, const unsigned char* d_bM, const float* d_arr, int useN,
	unsigned char* d_img, float* d_out,
	const float eye[3], const float u[3], const float v[3], const float w[3], float pixel) {
	int n = WIDTH * HEIGHT;
	K_RenderX << <(n + 255) / 256, 256 >> > (d_vol, d_bM, d_arr, useN, d_img, d_out,
		make_float3(eye[0], eye[1], eye[2]), make_float3(u[0], u[1], u[2]),
		make_float3(v[0], v[1], v[2]), make_float3(w[0], w[1], w[2]), pixel);
}


//==================== (비교 방법) 중앙차분 · 가우시안 평활 ====================

// 이진 볼륨 -> 실수 볼륨 (스레드 하나 = 복셀 하나)
__global__ void K_ToFloat(const unsigned char* vol, float* g, int nVox) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= nVox) return;
	g[idx] = (float)vol[idx];
}

// 한 축(axis 0 = x, 1 = y, 2 = z) 방향 가우시안 평활 : out = Σ_k w_k in(이웃 k), k = -R..R, w_k = exp(-k^2 / 2σ^2)
//  볼륨 밖 이웃은 0 (dPhi 가 볼륨 밖을 바깥(0)으로 보는 것과 같음). 가중치 합은 볼륨 밖 이웃까지 포함한 전체로 나눔
__global__ void K_Blur1D(const float* in, float* out, int nVox, int axis, int R, float sigma) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= nVox) return;
	int x = idx % VOLX;
	int y = (idx / VOLX) % VOLY;
	int z = idx / (VOLX * VOLY);
	int p[3] = { x, y, z };
	int DIM[3] = { VOLX, VOLY, VOLZ };
	int STEP[3] = { 1, VOLX, VOLX * VOLY };
	float sum = 0.0f, wsum = 0.0f;
	for (int k = -R; k <= R; k++) {
		float w = expf(-(float)(k * k) / (2.0f * sigma * sigma));
		wsum += w;
		int q = p[axis] + k;
		if (q < 0) continue;
		if (q > DIM[axis] - 1) continue;
		sum += w * in[idx + k * STEP[axis]];
	}
	out[idx] = sum / wsum;
}

// g 의 삼선형 보간. dIsOutside 와 같은 범위 밖이면 0
__device__ float dSampleG(const float* g, float px, float py, float pz) {
	if (dIsOutside(px, py, pz)) return 0.0f;
	int ix = (int)px, iy = (int)py, iz = (int)pz;
	float wx = px - ix, wy = py - iy, wz = pz - iz;
	int i000 = (iz * VOLY + iy) * VOLX + ix;
	int DX = 1, DY = VOLX, DZ = VOLX * VOLY;
	return g[i000] * (1 - wx) * (1 - wy) * (1 - wz)
		+ g[i000 + DX] * (wx) * (1 - wy) * (1 - wz)
		+ g[i000 + DY] * (1 - wx) * (wy) * (1 - wz)
		+ g[i000 + DX + DY] * (wx) * (wy) * (1 - wz)
		+ g[i000 + DZ] * (1 - wx) * (1 - wy) * (wz)
		+g[i000 + DZ + DX] * (wx) * (1 - wy) * (wz)
		+g[i000 + DZ + DY] * (1 - wx) * (wy) * (wz)
		+g[i000 + DZ + DX + DY] * (wx) * (wy) * (wz);
}

// 실시간 (비교 방법) : 광선·교점(이진 바이섹션)·조명은 K_Render 와 같음. 법선 = g 의 중앙차분 (간격 1 복셀)
//  out 이 0 이 아니면 픽셀마다 법선 3 + 교점 3 + 교점 여부 1 을 기록 (인공 데이터 정답 오차용)
__global__ void K_RenderG(const unsigned char* vol, const unsigned char* bM, const float* g, unsigned char* img, float* out,
	float3 eye3, float3 u3, float3 v3, float3 w3, float supersampling) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= WIDTH * HEIGHT) return;
	int x = id % WIDTH, y = id / WIDTH;
	float w[3] = { w3.x, w3.y, w3.z };
	if (out != 0) out[id * 7 + 6] = 0.0f;

	float RS[3];
	RS[0] = eye3.x + u3.x * (x - WIDTH * 0.5f) * supersampling + v3.x * (y - HEIGHT * 0.5f) * supersampling;
	RS[1] = eye3.y + u3.y * (x - WIDTH * 0.5f) * supersampling + v3.y * (y - HEIGHT * 0.5f) * supersampling;
	RS[2] = eye3.z + u3.z * (x - WIDTH * 0.5f) * supersampling + v3.z * (y - HEIGHT * 0.5f) * supersampling;

	float col[3] = { 0, 0, 0 };

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
		float phiBefore = dPhi(vol, RS[0] + w[0] * tm, RS[1] + w[1] * tm, RS[2] + w[2] * tm);
		for (float t = tm; t < tM; t = t + step) {
			float p[3] = { RS[0] + w[0] * t, RS[1] + w[1] * t, RS[2] + w[2] * t };
			if (dIsOutside(p[0], p[1], p[2])) continue;

			int bx = ((int)p[0]) >> BSHIFT, by = ((int)p[1]) >> BSHIFT, bz = ((int)p[2]) >> BSHIFT;
			if (bM[(bz * BY_COUNT + by) * BX_COUNT + bx] == 0) {
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
				phiBefore = dPhi(vol, RS[0] + w[0] * t, RS[1] + w[1] * t, RS[2] + w[2] * t);
				continue;
			}

			float phi = dPhi(vol, p[0], p[1], p[2]);
			if (phiBefore * phi < 0.0f || (phi == 0.0f && phiBefore != 0.0f)) {
				float a[3] = { RS[0] + w[0] * tBefore, RS[1] + w[1] * tBefore, RS[2] + w[2] * tBefore };
				float b[3] = { p[0], p[1], p[2] };
				if (phiBefore > 0.0f) {
					for (int k = 0; k < 3; k++) { float tmp = a[k]; a[k] = b[k]; b[k] = tmp; }
				}
				for (int it = 0; it < 10; it++) {
					float m[3] = { (a[0] + b[0]) * 0.5f, (a[1] + b[1]) * 0.5f, (a[2] + b[2]) * 0.5f };
					if (dPhi(vol, m[0], m[1], m[2]) < 0.0f) { a[0] = m[0]; a[1] = m[1]; a[2] = m[2]; }
					else { b[0] = m[0]; b[1] = m[1]; b[2] = m[2]; }
				}
				float hit[3] = { (a[0] + b[0]) * 0.5f, (a[1] + b[1]) * 0.5f, (a[2] + b[2]) * 0.5f };

				// (비교 방법) 법선 = g 의 중앙차분 그래디언트
				float N[3];
				N[0] = (dSampleG(g, hit[0] + 1, hit[1], hit[2]) - dSampleG(g, hit[0] - 1, hit[1], hit[2])) * 0.5f;
				N[1] = (dSampleG(g, hit[0], hit[1] + 1, hit[2]) - dSampleG(g, hit[0], hit[1] - 1, hit[2])) * 0.5f;
				N[2] = (dSampleG(g, hit[0], hit[1], hit[2] + 1) - dSampleG(g, hit[0], hit[1], hit[2] - 1)) * 0.5f;
				float len = sqrtf(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]);
				if (len > 0.0f) { N[0] /= len; N[1] /= len; N[2] /= len; }

				// 아래는 K_Render 와 같음
				float Vw[3] = { -w[0], -w[1], -w[2] };
				if (N[0] * Vw[0] + N[1] * Vw[1] + N[2] * Vw[2] < 0.0f) { N[0] = -N[0]; N[1] = -N[1]; N[2] = -N[2]; }

				if (out != 0) {
					out[id * 7 + 0] = N[0]; out[id * 7 + 1] = N[1]; out[id * 7 + 2] = N[2];
					out[id * 7 + 3] = hit[0]; out[id * 7 + 4] = hit[1]; out[id * 7 + 5] = hit[2];
					out[id * 7 + 6] = 1.0f;
				}

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

// (비교 방법) 창구 함수
void GpuToFloat(const unsigned char* d_vol, float* d_g) {
	int n = hVOLX * hVOLY * hVOLZ;
	K_ToFloat << <(n + 255) / 256, 256 >> > (d_vol, d_g, n);
}

void GpuBlur1D(const float* d_in, float* d_out, int axis, int R, float sigma) {
	int n = hVOLX * hVOLY * hVOLZ;
	K_Blur1D << <(n + 255) / 256, 256 >> > (d_in, d_out, n, axis, R, sigma);
}

void GpuRenderG(const unsigned char* d_vol, const unsigned char* d_bM, const float* d_g, unsigned char* d_img, float* d_out,
	const float eye[3], const float u[3], const float v[3], const float w[3], float pixel) {
	int n = WIDTH * HEIGHT;
	K_RenderG << <(n + 255) / 256, 256 >> > (d_vol, d_bM, d_g, d_img, d_out,
		make_float3(eye[0], eye[1], eye[2]), make_float3(u[0], u[1], u[2]),
		make_float3(v[0], v[1], v[2]), make_float3(w[0], w[1], w[2]), pixel);
}
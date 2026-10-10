// GPU 쪽 : 커널 함수 + main.cpp 가 부르는 창구 함수
#include <cuda_runtime.h>
#include <math.h>

// main.cpp 와 반드시 같은 값 (main.cpp 를 바꾸면 여기도 바꿀 것)
#define WIDTH   512
#define HEIGHT  512
#define VOLX 256
#define VOLY 256
#define VOLZ 225
const int BSIZE = 8;
const int BSHIFT = 3;
const int BY_COUNT = VOLY / BSIZE + (VOLY % BSIZE != 0);
const int BX_COUNT = VOLX / BSIZE + (VOLX % BSIZE != 0);

// 격자점 (x, y, z) 의 이진값 0 또는 1
__device__ int dVoxel(const unsigned char* vol, int x, int y, int z) {
	return vol[(z * VOLY + y) * VOLX + x];
}

// GetDensity 와 같은 삼선형 보간
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

// isOutside 와 같음
__device__ bool dIsOutside(float px, float py, float pz) {
	if (px >= VOLX - 1) return true;
	if (px < 0) return true;
	if (py >= VOLY - 1) return true;
	if (py < 0) return true;
	if (pz >= VOLZ - 1) return true;
	if (pz < 0) return true;
	return false;
}

// Phi 와 같음 : 안쪽 양수, 바깥 음수 (0.5 = ISO_LEVEL)
__device__ float dPhi(const unsigned char* vol, float px, float py, float pz) {
	if (dIsOutside(px, py, pz)) return 0.0f - 0.5f;
	return dDensity(vol, px, py, pz) - 0.5f;
}

// Jacobi 고유분해 : 끝나면 A 대각 = 고윳값, E 의 열 = 고유벡터
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

// 공분산 6성분 c 의 고유분해 : lam[0] >= lam[1] >= lam[2], o[k] = lam[k] 의 고유벡터가 있는 E 의 열 번호
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

// 1단계 : 격자점마다 정육면체 창 안 간선 중점으로 가우시안 가중 PCA -> 1차 공분산 (z 한 장, 스레드 하나 = 격자점 하나)
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

// 2단계 : 1차 법선 n0 로 눕힌 3시그마 타원체 안 간선 중점으로 가중 PCA -> 2차 공분산 (z 한 장)
__global__ void K_Step2(const unsigned char* vol, const float* covA, float* covB, int z,
	float sn, float stMax, float rhoFallback, float rhoScale) {
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

	float lt = (lam[0] + lam[1]) * 0.5f;
	int fallback = 0;
	float rho = 0.0f;
	if (lt <= 0.0f) fallback = 1;
	else {
		rho = lam[2] / lt;
		if (rho >= rhoFallback) fallback = 1;
	}
	if (fallback == 1) {
		for (int k = 0; k < 6; k++) B[k] = A[k];
		return;
	}

	//float s = 1.0f / (1.0f + (rho / rhoScale) * (rho / rhoScale));
	//float st = fmaxf(stMax * s, sn);
	float st = stMax;
	float invSn2 = 1.0f / (sn * sn);
	float invSt2 = 1.0f / (st * st);

	int xq[3] = { x, y, z };
	int DIM[3] = { VOLX, VOLY, VOLZ };
	float e[3];
	for (int k = 0; k < 3; k++)
		e[k] = 3.0f * sqrtf(st * st + (sn * sn - st * st) * n0[k] * n0[k]);

	float W = 0, M[3] = { 0, 0, 0 };
	float Sxx = 0, Syy = 0, Szz = 0, Sxy = 0, Sxz = 0, Syz = 0;
	int cnt = 0;

	for (int a = 0; a < 3; a++) {
		int b = (a + 1) % 3, c = (a + 2) % 3;
		int Kb = (int)floorf(e[b]);
		int Kc = (int)floorf(e[c]);
		int i0 = (int)floorf(xq[a] - e[a]);
		int i1 = (int)floorf(xq[a] + e[a]);
		for (int pb = xq[b] - Kb; pb <= xq[b] + Kb; pb++)
			for (int pc = xq[c] - Kc; pc <= xq[c] + Kc; pc++) {
				if (pb < 0) continue; if (pb >= DIM[b] - 1) continue;
				if (pc < 0) continue; if (pc >= DIM[c] - 1) continue;

				for (int i = i0; i <= i1; i++) {
					if (i < 0) continue; if (i + 2 > DIM[a] - 1) continue;
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
					cnt++;
				}
			}
	}
	/*if (cnt == 0) {
		for (int k = 0; k < 6; k++) B[k] = A[k];
		return;
	}*/

	float iW = 1.0f / W;
	float px = M[0] * iW, py = M[1] * iW, pz = M[2] * iW;
	B[0] = Sxx * iW - px * px;
	B[1] = Syy * iW - py * py;
	B[2] = Szz * iW - pz * pz;
	B[3] = Sxy * iW - px * py;
	B[4] = Sxz * iW - px * pz;
	B[5] = Syz * iW - py * pz;
}

// 실시간 : 광선 전진 -> 이진 바이섹션 교점 -> 8이웃 2차 공분산을 삼선형 가중합 (확신 없음) -> 최소 고유벡터 법선 -> 조명 (스레드 하나 = 픽셀 하나)
__global__ void K_Render(const unsigned char* vol, const unsigned char* bM, const float* covB, unsigned char* img,
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
			if (phiBefore * phi < 0.0f) {
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

							float g = tri;              // v1-2 : 확신 없음, 삼선형만
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

// 1단계 실행 (z 한 장)
void GpuStep1(const unsigned char* d_vol, float* d_A, int z, int R, float sigma) {
	int n = VOLX * VOLY;
	K_Step1 << <(n + 255) / 256, 256 >> > (d_vol, d_A, z, R, sigma);
}

// 2단계 실행 (z 한 장)
void GpuStep2(const unsigned char* d_vol, const float* d_A, float* d_B, int z,
	float sn, float stMax, float rhoFallback, float rhoScale) {
	int n = VOLX * VOLY;
	K_Step2 << <(n + 255) / 256, 256 >> > (d_vol, d_A, d_B, z, sn, stMax, rhoFallback, rhoScale);
}

// 렌더 실행 (화면 전체)
void GpuRender(const unsigned char* d_vol, const unsigned char* d_bM, const float* d_B, unsigned char* d_img,
	const float eye[3], const float u[3], const float v[3], const float w[3]) {
	int n = WIDTH * HEIGHT;
	K_Render << <(n + 255) / 256, 256 >> > (d_vol, d_bM, d_B, d_img,
		make_float3(eye[0], eye[1], eye[2]), make_float3(u[0], u[1], u[2]),
		make_float3(v[0], v[1], v[2]), make_float3(w[0], w[1], w[2]));
}
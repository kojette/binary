// =====================================================================
// main.cpp  -  호스트 쪽 전부
//
// 하는 일 셋뿐이다.
//   1. CPU 로 한 판 그린다 (지금까지의 코드와 같은 로직, 지표는 전부 뺐다)
//   2. GPU 로 같은 것을 그린다 (손 보간판과 텍스처판 둘 다)
//   3. 단계별 시간을 나란히 놓고, 그림이 같은지 화소로 확인한다
//
// 그림이 다르면 시간은 의미가 없다. 그래서 일치율을 먼저 찍는다.
// =====================================================================
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <vector>
#include "common.h"

using namespace std;
using Clock = chrono::high_resolution_clock;
static double Ms(Clock::time_point a, Clock::time_point b) {
    return chrono::duration_cast<chrono::microseconds>(b - a).count() * 0.001;
}

// =====================================================================
// 1. 자료
// =====================================================================
static unsigned char* g_vol;        // 원본 밀도
static unsigned char* g_bin;        // 이진 0 또는 255
static unsigned char* g_blockMax;
static vector<int>    g_bandIdx;
static vector<float>  g_gField;
static vector<float>  g_bandCov;
static int            g_nBand = 1;

static unsigned char* g_imgCpu;
static unsigned char* g_imgGpuHand;
static unsigned char* g_imgGpuTex;

// =====================================================================
// 2. CPU 기본 함수. GPU 커널과 한 글자도 다르지 않게 맞춰 둔다
// =====================================================================
static inline float BinAt(int x, int y, int z) {
    if (x < 0 || x >= VOLX || y < 0 || y >= VOLY || z < 0 || z >= VOLZ) return 0.0f;
    return g_bin[(z * VOLY + y) * VOLX + x] * (1.0f / 255.0f);
}

static float Density(float px, float py, float pz) {
    int ix = (int)floorf(px), iy = (int)floorf(py), iz = (int)floorf(pz);
    float wx = px - ix, wy = py - iy, wz = pz - iz;
    float d = 0.0f;
    for (int k = 0; k < 8; k++) {
        int dx = k & 1, dy = (k >> 1) & 1, dz = (k >> 2) & 1;
        float fx = wx * dx + (1.0f - wx) * (1 - dx);
        float fy = wy * dy + (1.0f - wy) * (1 - dy);
        float fz = wz * dz + (1.0f - wz) * (1 - dz);
        d += fx * fy * fz * BinAt(ix + dx, iy + dy, iz + dz);
    }
    return d;
}

struct V3 { float x, y, z; };
static inline V3 mk(float a, float b, float c) { V3 v; v.x = a; v.y = b; v.z = c; return v; }

static inline float Phi(V3 p) { return Density(p.x, p.y, p.z) - ISO_BIN; }

static V3 Bisect(V3 a, V3 b) {
    for (int i = 0; i < 8; i++) {
        V3 m = mk((a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f, (a.z + b.z) * 0.5f);
        if (Phi(m) < 0.0f) a = m; else b = m;
    }
    return mk((a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f, (a.z + b.z) * 0.5f);
}

static V3 GradientAt(float px, float py, float pz) {
    V3 g = mk(0, 0, 0);
    const float inv2s2 = 1.0f / (2.0f * GRAD_SIG * GRAD_SIG);
    for (int oz = -GRAD_R; oz <= GRAD_R; oz++)
        for (int oy = -GRAD_R; oy <= GRAD_R; oy++)
            for (int ox = -GRAD_R; ox <= GRAD_R; ox++) {
                int zero = (ox == 0) * (oy == 0) * (oz == 0);
                float r2 = (float)(ox * ox + oy * oy + oz * oz);
                float w = expf(-r2 * inv2s2) * (1 - zero);
                float d = Density(px + ox, py + oy, pz + oz);
                g.x += w * ox * d;  g.y += w * oy * d;  g.z += w * oz * d;
            }
    return g;
}

// =====================================================================
// 3. CPU 전처리 세 단계. 각각 따로 잰다
// =====================================================================
static void CpuBlockMax() {
    for (int b = 0; b < BLOCKN; b++) {
        int bx = b % BX_COUNT;
        int by = (b / BX_COUNT) % BY_COUNT;
        int bz = b / (BX_COUNT * BY_COUNT);
        unsigned char m = 0;
        for (int z = bz * BSIZE; z < bz * BSIZE + BSIZE && z < VOLZ; z++)
            for (int y = by * BSIZE; y < by * BSIZE + BSIZE && y < VOLY; y++)
                for (int x = bx * BSIZE; x < bx * BSIZE + BSIZE && x < VOLX; x++) {
                    unsigned char v = g_vol[(z * VOLY + y) * VOLX + x];
                    m = (v > m) ? v : m;
                }
        g_blockMax[b] = m;
    }
}

static void CpuShell() {
    g_bandIdx.assign(VOLN, 0);
    g_nBand = 1;                       // 0번은 "껍질 아님" 자리로 비워 둔다
    for (int z = 1; z < VOLZ - 1; z++)
        for (int y = 1; y < VOLY - 1; y++)
            for (int x = 1; x < VOLX - 1; x++) {
                int i = (z * VOLY + y) * VOLX + x;
                int s = (g_bin[i] > 127)
                    + (g_bin[i - 1] > 127) + (g_bin[i + 1] > 127)
                    + (g_bin[i - VOLX] > 127) + (g_bin[i + VOLX] > 127)
                    + (g_bin[i - VOLX * VOLY] > 127) + (g_bin[i + VOLX * VOLY] > 127);
                if (s * (7 - s) != 0) g_bandIdx[i] = g_nBand++;
            }
}

static void CpuGField() {
    g_gField.assign((size_t)g_nBand * 3, 0.0f);
    for (int z = 1; z < VOLZ - 1; z++)
        for (int y = 1; y < VOLY - 1; y++)
            for (int x = 1; x < VOLX - 1; x++) {
                int bi = g_bandIdx[(z * VOLY + y) * VOLX + x];
                if (!bi) continue;
                V3 g = GradientAt((float)x, (float)y, (float)z);
                g_gField[bi * 3 + 0] = g.x;
                g_gField[bi * 3 + 1] = g.y;
                g_gField[bi * 3 + 2] = g.z;
            }
}

static void CpuBandCov() {
    g_bandCov.assign((size_t)g_nBand * 6, 0.0f);
    const float inv2s2 = 1.0f / (2.0f * COV_SIG * COV_SIG);
    for (int z = COV_R + 1; z < VOLZ - COV_R - 1; z++)
        for (int y = COV_R + 1; y < VOLY - COV_R - 1; y++)
            for (int x = COV_R + 1; x < VOLX - COV_R - 1; x++) {
                int bi = g_bandIdx[(z * VOLY + y) * VOLX + x];
                if (!bi) continue;
                float c[6] = { 0,0,0,0,0,0 };
                for (int oz = -COV_R; oz <= COV_R; oz++)
                    for (int oy = -COV_R; oy <= COV_R; oy++)
                        for (int ox = -COV_R; ox <= COV_R; ox++) {
                            float r2 = (float)(ox * ox + oy * oy + oz * oz);
                            float w = expf(-r2 * inv2s2);
                            int nb = g_bandIdx[((z + oz) * VOLY + (y + oy)) * VOLX + (x + ox)];
                            float gx = g_gField[nb * 3 + 0];
                            float gy = g_gField[nb * 3 + 1];
                            float gz = g_gField[nb * 3 + 2];
                            c[0] += w * gx * gx;  c[1] += w * gy * gy;  c[2] += w * gz * gz;
                            c[3] += w * gx * gy;  c[4] += w * gx * gz;  c[5] += w * gy * gz;
                        }
                for (int k = 0; k < 6; k++) g_bandCov[bi * 6 + k] = c[k];
            }
}

// =====================================================================
// 4. CPU 텐서 보간과 고유해법. 커널과 같은 식이다
// =====================================================================
static void FetchCov(V3 p, float C[6]) {
    int ix = (int)p.x, iy = (int)p.y, iz = (int)p.z;
    if (ix < 0) ix = 0;  if (ix > VOLX - 2) ix = VOLX - 2;
    if (iy < 0) iy = 0;  if (iy > VOLY - 2) iy = VOLY - 2;
    if (iz < 0) iz = 0;  if (iz > VOLZ - 2) iz = VOLZ - 2;
    float wx = p.x - ix, wy = p.y - iy, wz = p.z - iz;
    C[0] = C[1] = C[2] = C[3] = C[4] = C[5] = 0.0f;
    for (int k = 0; k < 8; k++) {
        int dx = k & 1, dy = (k >> 1) & 1, dz = (k >> 2) & 1;
        float fx = wx * dx + (1.0f - wx) * (1 - dx);
        float fy = wy * dy + (1.0f - wy) * (1 - dy);
        float fz = wz * dz + (1.0f - wz) * (1 - dz);
        float w = fx * fy * fz;
        int b = g_bandIdx[((iz + dz) * VOLY + (iy + dy)) * VOLX + (ix + dx)] * 6;
        for (int m = 0; m < 6; m++) C[m] += w * g_bandCov[b + m];
    }
}

static V3 EigenAxis(const float C[6], V3 n) {
    float c0 = C[0], c1 = C[1], c2 = C[2], c3 = C[3], c4 = C[4], c5 = C[5];
    float A = c0 + c1 + c2;
    float B = c0 * c1 + c1 * c2 + c2 * c0 - c3 * c3 - c4 * c4 - c5 * c5;
    float D = c0 * c1 * c2 + 2.0f * c3 * c4 * c5
        - c0 * c5 * c5 - c1 * c4 * c4 - c2 * c3 * c3;
    float p = A * A / 3.0f - B;
    float lam;
    if (p <= 0.0f) lam = A / 3.0f;
    else {
        float q = A * A * A / 13.5f - A * B / 3.0f + D;
        float arg = q / (2.0f * powf(p / 3.0f, 1.5f));
        if (arg < -1.0f) arg = -1.0f;  if (arg > 1.0f) arg = 1.0f;
        lam = A / 3.0f + 2.0f * sqrtf(p / 3.0f) * cosf(acosf(arg) / 3.0f);
    }
    V3 r1 = mk(c0 - lam, c3, c4), r2 = mk(c3, c1 - lam, c5), r3 = mk(c4, c5, c2 - lam);
    V3 v1 = mk(r1.y * r2.z - r1.z * r2.y, r1.z * r2.x - r1.x * r2.z, r1.x * r2.y - r1.y * r2.x);
    V3 v2 = mk(r2.y * r3.z - r2.z * r3.y, r2.z * r3.x - r2.x * r3.z, r2.x * r3.y - r2.y * r3.x);
    V3 v3 = mk(r3.y * r1.z - r3.z * r1.y, r3.z * r1.x - r3.x * r1.z, r3.x * r1.y - r3.y * r1.x);
    float l1 = v1.x * v1.x + v1.y * v1.y + v1.z * v1.z;
    float l2 = v2.x * v2.x + v2.y * v2.y + v2.z * v2.z;
    float l3 = v3.x * v3.x + v3.y * v3.y + v3.z * v3.z;
    V3 best = v1; float bl = l1;
    if (l2 > bl) { best = v2; bl = l2; }
    if (l3 > bl) { best = v3; bl = l3; }
    if (bl < 1e-24f) {
        float ln = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z) + 1e-20f;
        return mk(n.x / ln, n.y / ln, n.z / ln);
    }
    float inv = 1.0f / sqrtf(bl);
    return mk(best.x * inv, best.y * inv, best.z * inv);
}

static V3 Orient(V3 p, V3 N, V3 w) {
    V3 a = mk(p.x + N.x * 0.5f, p.y + N.y * 0.5f, p.z + N.z * 0.5f);
    V3 b = mk(p.x - N.x * 0.5f, p.y - N.y * 0.5f, p.z - N.z * 0.5f);
    float d = Phi(a) - Phi(b);
    float s1 = (d > 0.0f) ? -1.0f : 1.0f;
    N = mk(N.x * s1, N.y * s1, N.z * s1);
    float dot = N.x * w.x + N.y * w.y + N.z * w.z;
    float s2 = (fabsf(d) < 1e-4f && dot > 0.0f) ? -1.0f : 1.0f;
    return mk(N.x * s2, N.y * s2, N.z * s2);
}

// =====================================================================
// 5. CPU 렌더. GPU 커널과 같은 순서로 쓴다
// =====================================================================
static void CpuRender(int mode, unsigned char* img) {
    memset(img, 0, NPIX * 3);

    V3 eye = mk(EYE_X, EYE_Y, EYE_Z);
    V3 wv = mk(AT_X - eye.x, AT_Y - eye.y, AT_Z - eye.z);
    float wl = 1.0f / sqrtf(wv.x * wv.x + wv.y * wv.y + wv.z * wv.z);
    wv = mk(wv.x * wl, wv.y * wl, wv.z * wl);
    V3 up = mk(0, 1, 0);
    V3 uv = mk(up.y * wv.z - up.z * wv.y, up.z * wv.x - up.x * wv.z, up.x * wv.y - up.y * wv.x);
    float ul = 1.0f / sqrtf(uv.x * uv.x + uv.y * uv.y + uv.z * uv.z);
    uv = mk(uv.x * ul, uv.y * ul, uv.z * ul);
    V3 vv = mk(wv.y * uv.z - wv.z * uv.y, wv.z * uv.x - wv.x * uv.z, wv.x * uv.y - wv.y * uv.x);

    for (int py = 0; py < HEIGHT; py++)
        for (int px = 0; px < WIDTH; px++) {
            float fx = (px - WIDTH * 0.5f) * SS;
            float fy = (py - HEIGHT * 0.5f) * SS;
            V3 RS = mk(eye.x + uv.x * fx + vv.x * fy,
                eye.y + uv.y * fx + vv.y * fy,
                eye.z + uv.z * fx + vv.z * fy);

            float t1, t2;
            t1 = -RS.x / wv.x;  t2 = ((VOLX - 1) - RS.x) / wv.x;
            float xm = fminf(t1, t2), xM = fmaxf(t1, t2);
            t1 = -RS.y / wv.y;  t2 = ((VOLY - 1) - RS.y) / wv.y;
            float ym = fminf(t1, t2), yM = fmaxf(t1, t2);
            t1 = -RS.z / wv.z;  t2 = ((VOLZ - 1) - RS.z) / wv.z;
            float zm = fminf(t1, t2), zM = fmaxf(t1, t2);
            float tm = fmaxf(fmaxf(xm, ym), zm);
            float tM = fminf(fminf(xM, yM), zM);
            if (tm >= tM) continue;

            float tBefore = tm;
            float phiBefore = Phi(mk(RS.x + wv.x * tm, RS.y + wv.y * tm, RS.z + wv.z * tm));

            for (float t = tm; t < tM; t += STEP) {
                V3 p = mk(RS.x + wv.x * t, RS.y + wv.y * t, RS.z + wv.z * t);
                if (p.x < 0 || p.x >= VOLX - 1 || p.y < 0 || p.y >= VOLY - 1
                    || p.z < 0 || p.z >= VOLZ - 1) continue;

                int bx = ((int)p.x) >> BSHIFT, by = ((int)p.y) >> BSHIFT, bz = ((int)p.z) >> BSHIFT;
                int bid = (bz * BY_COUNT + by) * BX_COUNT + bx;

                if (g_blockMax[bid] < ISO) {
                    float jump = 0.0f; int nb;
                    do {
                        jump += 1.0f;
                        V3 q = mk(p.x + wv.x * jump, p.y + wv.y * jump, p.z + wv.z * jump);
                        int qx = ((int)q.x) >> BSHIFT, qy = ((int)q.y) >> BSHIFT, qz = ((int)q.z) >> BSHIFT;
                        nb = (qz * BY_COUNT + qy) * BX_COUNT + qx;
                    } while (nb == bid && jump < 64.0f);
                    t += (jump - STEP);
                    tBefore = t;
                    phiBefore = Phi(mk(RS.x + wv.x * t, RS.y + wv.y * t, RS.z + wv.z * t));
                    continue;
                }

                float phi = Phi(p);
                if (phiBefore * phi < 0.0f) {
                    V3 pB = mk(RS.x + wv.x * tBefore, RS.y + wv.y * tBefore, RS.z + wv.z * tBefore);
                    V3 hit = (phiBefore < 0.0f) ? Bisect(pB, p) : Bisect(p, pB);

                    V3 N;
                    if (mode == MODE_COV) {
                        float C[6]; FetchCov(hit, C);
                        if (C[0] + C[1] + C[2] < 1e-12f) {
                            V3 g = GradientAt(hit.x, hit.y, hit.z);
                            float l = 1.0f / (sqrtf(g.x * g.x + g.y * g.y + g.z * g.z) + 1e-20f);
                            N = mk(-g.x * l, -g.y * l, -g.z * l);
                        }
                        else {
                            N = EigenAxis(C, mk(0.577f, 0.577f, 0.577f));
                            N = Orient(hit, N, wv);
                        }
                    }
                    else {
                        V3 g = GradientAt(hit.x, hit.y, hit.z);
                        float l = 1.0f / (sqrtf(g.x * g.x + g.y * g.y + g.z * g.z) + 1e-20f);
                        N = mk(-g.x * l, -g.y * l, -g.z * l);
                    }

                    V3 L = mk(-wv.x, -wv.y + 0.3f, -wv.z);
                    float ll = 1.0f / sqrtf(L.x * L.x + L.y * L.y + L.z * L.z);
                    L = mk(L.x * ll, L.y * ll, L.z * ll);
                    float NL = fabsf(N.x * L.x + N.y * L.y + N.z * L.z);

                    float r = 0.15f * 0.9f * 0.8f + 0.85f * 0.90f * NL;
                    float g2 = 0.15f * 0.85f * 0.8f + 0.85f * 0.85f * NL;
                    float b = 0.15f * 0.80f * 0.8f + 0.85f * 0.80f * NL;

                    int o = (py * WIDTH + px) * 3;
                    img[o + 0] = (unsigned char)(fminf(r, 1.0f) * 255.0f);
                    img[o + 1] = (unsigned char)(fminf(g2, 1.0f) * 255.0f);
                    img[o + 2] = (unsigned char)(fminf(b, 1.0f) * 255.0f);
                    break;
                }
                tBefore = t;
                phiBefore = phi;
            }
        }
}

// =====================================================================
// 6. 그림 저장과 비교
// =====================================================================
static void SaveBMP(const char* name, const unsigned char* img) {
    const int rowSize = WIDTH * 3;
    const int dataSize = rowSize * HEIGHT;
    unsigned char fh[14] = { 'B','M' }, ih[40] = { 0 };
    *(int*)&fh[2] = 54 + dataSize;  *(int*)&fh[10] = 54;
    *(int*)&ih[0] = 40;  *(int*)&ih[4] = WIDTH;  *(int*)&ih[8] = HEIGHT;
    *(short*)&ih[12] = 1;  *(short*)&ih[14] = 24;  *(int*)&ih[20] = dataSize;
    FILE* f = fopen(name, "wb");
    if (!f) { printf("저장 실패 %s\n", name); return; }
    fwrite(fh, 1, 14, f);  fwrite(ih, 1, 40, f);
    for (int y = 0; y < HEIGHT; y++)
        for (int x = 0; x < WIDTH; x++) {
            const unsigned char* q = img + (y * WIDTH + x) * 3;
            unsigned char bgr[3] = { q[2], q[1], q[0] };
            fwrite(bgr, 1, 3, f);
        }
    fclose(f);
}

// 두 그림이 같은지. 시간을 믿으려면 그림이 먼저 같아야 한다
static void Compare(const char* label, const unsigned char* a, const unsigned char* b) {
    long same = 0, diff1 = 0, diffBig = 0, obj = 0;
    double sum = 0.0;
    for (int i = 0; i < NPIX; i++) {
        int d = 0;
        for (int k = 0; k < 3; k++) {
            int e = (int)a[i * 3 + k] - (int)b[i * 3 + k];
            if (e < 0) e = -e;
            if (e > d) d = e;
        }
        if (a[i * 3] > 3 || b[i * 3] > 3) obj++;
        sum += d;
        if (d == 0) same++;
        else if (d <= 1) diff1++;
        else diffBig++;
    }
    printf("  %-22s 완전일치 %6.2f%%   1단계차 %5.2f%%   2이상차 %5.2f%%   평균차 %.3f\n",
        label, 100.0 * same / NPIX, 100.0 * diff1 / NPIX, 100.0 * diffBig / NPIX,
        sum / NPIX);
}

// =====================================================================
// 7. 표 출력
// =====================================================================
static void PrintRow(const char* name, double cpu, double gpu) {
    if (cpu <= 0.0 && gpu <= 0.0) return;
    double ratio = (gpu > 0.0001) ? cpu / gpu : 0.0;
    printf("  %-16s %10.2f %10.2f   ", name, cpu, gpu);
    if (cpu <= 0.0) printf("     -      (CPU 에는 없는 단계)\n");
    else if (ratio >= 1.0) printf("%7.1f 배 빠름\n", ratio);
    else printf("%7.2f 배 느림\n", 1.0 / (ratio + 1e-9));
}

static void PrintTable(const char* title, const Timing& c, const Timing& g) {
    double ct = c.blocks + c.shell + c.gfield + c.cov + c.render;
    double gt = g.upload + g.blocks + g.shell + g.gfield + g.cov + g.render + g.download;
    printf("\n=== %s ===\n", title);
    printf("  %-16s %10s %10s   %s\n", "단계", "CPU ms", "GPU ms", "배율");
    PrintRow("업로드", 0.0, g.upload);
    PrintRow("블록최대값", c.blocks, g.blocks);
    PrintRow("껍질색인", c.shell, g.shell);
    PrintRow("기울기장", c.gfield, g.gfield);
    PrintRow("텐서누적", c.cov, g.cov);
    PrintRow("렌더", c.render, g.render);
    PrintRow("내려받기", 0.0, g.download);
    printf("  %-16s %10.2f %10.2f   %7.1f 배 빠름\n", "합계", ct, gt, ct / (gt + 1e-9));

    // 어느 단계를 옮긴 것이 가장 이득이었나. 절감 시간으로 줄을 세운다
    struct S { const char* n; double save; } s[5] = {
        {"블록최대값", c.blocks - g.blocks},
        {"껍질색인",   c.shell - g.shell},
        {"기울기장",   c.gfield - g.gfield},
        {"텐서누적",   c.cov - g.cov},
        {"렌더",       c.render - g.render},
    };
    for (int i = 0; i < 5; i++)
        for (int j = i + 1; j < 5; j++)
            if (s[j].save > s[i].save) { S tmp = s[i]; s[i] = s[j]; s[j] = tmp; }
    printf("  절감 큰 순서 : ");
    for (int i = 0; i < 5; i++)
        printf("%s %.0fms%s", s[i].n, s[i].save, (i < 4) ? " > " : "\n");
    printf("  전체 대비 비중 : ");
    for (int i = 0; i < 5; i++)
        printf("%s %.1f%%%s", s[i].n, 100.0 * s[i].save / (ct - gt + 1e-9),
            (i < 4) ? " " : "\n");
}

// =====================================================================
// 8. main
// =====================================================================
int main(void) {
    g_vol = (unsigned char*)malloc(VOLN);
    g_bin = (unsigned char*)malloc(VOLN);
    g_blockMax = (unsigned char*)malloc(BLOCKN);
    g_imgCpu = (unsigned char*)malloc(NPIX * 3);
    g_imgGpuHand = (unsigned char*)malloc(NPIX * 3);
    g_imgGpuTex = (unsigned char*)malloc(NPIX * 3);

    FILE* f = fopen("bighead.den", "rb");
    if (!f) { printf("bighead.den 을 못 찾았습니다\n"); return 1; }
    fread(g_vol, 1, VOLN, f);
    fclose(f);
    for (int i = 0; i < VOLN; i++) g_bin[i] = (g_vol[i] >= ISO) ? 255 : 0;

    printf("볼륨 %dx%dx%d = %.1f MB,  화면 %dx%d\n",
        VOLX, VOLY, VOLZ, VOLN / 1048576.0, WIDTH, HEIGHT);
    GpuPrintDevice();

    // ---------------- 두 설정을 차례로 ----------------
    const int modes[2] = { MODE_GRADIENT, MODE_COV };
    const char* modeName[2] = { "가중이웃 (기준선, 전처리 없음)",
                                "공분산 저장후보간 (정석)" };

    for (int m = 0; m < 2; m++) {
        int mode = modes[m];
        Timing c = { 0,0,0,0,0,0,0 }, gH = { 0 }, gT = { 0 };
        double memH = 0.0, memT = 0.0;

        // ---- CPU ----
        auto t0 = Clock::now();  CpuBlockMax();      auto t1 = Clock::now();
        CpuShell();                                  auto t2 = Clock::now();
        if (mode == MODE_COV) CpuGField();           auto t3 = Clock::now();
        if (mode == MODE_COV) CpuBandCov();          auto t4 = Clock::now();

        // 렌더는 3회 중 최소값. 같은 계산이 10% 씩 흔들리는 것을 이미 봤다
        double best = 1e18;
        for (int k = 0; k < 3; k++) {
            auto a = Clock::now();  CpuRender(mode, g_imgCpu);  auto b = Clock::now();
            double v = Ms(a, b);  if (v < best) best = v;
        }
        c.blocks = Ms(t0, t1);  c.shell = Ms(t1, t2);
        c.gfield = Ms(t2, t3);  c.cov = Ms(t3, t4);  c.render = best;

        double cpuMem = (VOLN + BLOCKN + (double)VOLN * 4
            + (mode == MODE_COV ? (double)g_nBand * 9 * 4 : 0.0)) / 1048576.0;

        // ---- GPU 두 벌 ----
        GpuRender(g_vol, mode, FETCH_HAND, g_imgGpuHand, &gH, &memH);
        GpuRender(g_vol, mode, FETCH_HAND, g_imgGpuHand, &gH, &memH);   // 첫 호출은 예열
        GpuRender(g_vol, mode, FETCH_TEX, g_imgGpuTex, &gT, &memT);
        GpuRender(g_vol, mode, FETCH_TEX, g_imgGpuTex, &gT, &memT);

        printf("\n\n################ %s ################\n", modeName[m]);
        printf("[그림 일치] 시간을 믿으려면 그림이 먼저 같아야 합니다\n");
        Compare("GPU 손보간 대 CPU", g_imgGpuHand, g_imgCpu);
        Compare("GPU 텍스처 대 CPU", g_imgGpuTex, g_imgCpu);
        printf("  손보간판은 CPU 와 완전일치가 나와야 정상입니다.\n");
        printf("  텍스처판은 하드웨어 보간 가중치가 8비트라 조금 어긋납니다. 그게 대가입니다.\n");

        PrintTable("기초판 (손 삼선형 보간, CPU 와 같은 계산)", c, gH);
        PrintTable("최적화판 (3D 텍스처 하드웨어 보간)", c, gT);

        double gtH = gH.upload + gH.blocks + gH.shell + gH.gfield + gH.cov + gH.render + gH.download;
        double gtT = gT.upload + gT.blocks + gT.shell + gT.gfield + gT.cov + gT.render + gT.download;
        printf("\n  [기초판 대 최적화판] 렌더 %.2f ms 대 %.2f ms  (%.2f 배)   합계 %.2f 대 %.2f\n",
            gH.render, gT.render, gH.render / (gT.render + 1e-9), gtH, gtT);
        printf("  [메모리] CPU %.2f MB   GPU %.2f MB   원본 볼륨 %.1f MB\n",
            cpuMem, memH, VOLN / 1048576.0);

        char nm[128];
        sprintf(nm, "out_cpu_mode%d.bmp", mode);   SaveBMP(nm, g_imgCpu);
        sprintf(nm, "out_gpu_hand_mode%d.bmp", mode); SaveBMP(nm, g_imgGpuHand);
        sprintf(nm, "out_gpu_tex_mode%d.bmp", mode);  SaveBMP(nm, g_imgGpuTex);
    }

    printf("\n읽는 법\n");
    printf("  1. 그림 일치가 먼저다. 손보간판이 CPU 와 다르면 아래 숫자는 전부 못 쓴다\n");
    printf("  2. 배율이 큰 단계가 GPU 로 옮겨 이득 본 단계다\n");
    printf("  3. 절감 큰 순서는 배율이 아니라 실제로 줄어든 밀리초로 줄을 세운 것이다.\n");
    printf("     배율 100배라도 원래 1ms 였으면 이득은 1ms 뿐이다 (암달의 법칙)\n");
    printf("  4. 업로드와 내려받기는 CPU 에 없던 비용이다. 이득에서 빼야 한다\n");

    return 0;
}
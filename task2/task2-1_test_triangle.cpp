// Created by sway on 2018/8/25.

/* 实现线性三角化方法(Linear triangulation methods), 给定匹配点
 * 以及相机投影矩阵(至少2对），计算对应的三维点坐标。给定相机内外参矩阵时，
 * 图像上每个点实际上对应三维中一条射线，理想情况下，利用两条射线相交便可以
 * 得到三维点的坐标。但是实际中，由于计算或者检测误差，无法保证两条射线的
 * 相交性，因此需要建立新的数学模型（如最小二乘）进行求解。
 *
 * 考虑两个视角的情况，假设空间中的三维点P的齐次坐标为X=[x,y,z,1]',对应地在
 * 两个视角的投影点分别为p1和p2，它们的图像坐标为
 *          x1=[x1, y1, 1]', x2=[x2, y2, 1]'.
 *
 * 两幅图像对应的相机投影矩阵为P1, P2 (P1,P2维度是3x4),理想情况下
 *             x1=P1X, x2=P2X
 *
 * 考虑第一个等式，在其两侧分别叉乘x1,可以得到
 *             x1 x (P1X) = 0
 *
 * 将P1X表示成[P11X, P21X, P31X]',其中P11，P21，P31分别是投影矩阵P1的第
 * 1～3行，我们可以得到
 *
 *          x1(P13X) - P11X     = 0
 *          y1(P13X) - P12X     = 0
 *          x1(P12X) - y1(P11X) = 0
 * 其中第三个方程可以由前两个通过线性变换得到，因此我们只考虑全两个方程。每一个
 * 视角可以提供两个约束，联合第二个视角的约束，我们可以得到
 *
 *                   AX = 0,
 * 其中
 *           [x1P13 - P11]
 *       A = [y1P13 - P12]
 *           [x2P23 - P21]
 *           [y2P23 - P22]
 *
 * 当视角个数多于2个的时候，可以采用最小二乘的方式进行求解，理论上，在不存在外点的
 * 情况下，视角越多估计的三维点坐标越准确。当存在外点(错误的匹配点）时，则通常采用
 * RANSAC的鲁棒估计方法进行求解。
 */

#include <math/matrix_svd.h>
#include "math/vector.h"
#include "math/matrix.h"

int main(int argc, char *argv[])
{

    /* change*/
    math::Vec2f p1;     //两个像平面点
    p1[0] = 0.289986;   //x1
    p1[1] = -0.0355493; //y1
    math::Vec2f p2;
    p2[0] = 0.316154;  //x2
    p2[1] = 0.0898488; //y2

    math::Matrix<double, 3, 4> P1, P2;
    //             [Pi1]
    //Pi=Ki[Ri,ti]=[Pi2]
    //             [Pi3]
    P1(0, 0) = 0.919653; //P11
    P1(0, 1) = -0.000621866;
    P1(0, 2) = -0.00124006;
    P1(0, 3) = 0.00255933;

    P1(1, 0) = 0.000609954; //P12
    P1(1, 1) = 0.919607;
    P1(1, 2) = -0.00957316;
    P1(1, 3) = 0.0540753;

    P1(2, 0) = 0.00135482; //P13
    P1(2, 1) = 0.0104087;
    P1(2, 2) = 0.999949;
    P1(2, 3) = -0.127624;

    P2(0, 0) = 0.920039; //P21
    P2(0, 1) = -0.0117214;
    P2(0, 2) = 0.0144298;
    P2(0, 3) = 0.0749395;

    P2(1, 0) = 0.0118301; //P22
    P2(1, 1) = 0.920129;
    P2(1, 2) = -0.00678373;
    P2(1, 3) = 0.862711;

    P2(2, 0) = -0.0155846; //P23
    P2(2, 1) = 0.00757181;
    P2(2, 2) = 0.999854;
    P2(2, 3) = -0.0887441;

    /* 构造A矩阵 */
    math::Matrix<double, 4, 4> A;
    /*
     * TODO 对A矩阵进行赋值
     */
    A(0, 0) = p1[0] * P1(2, 0) - P1(0, 0);
    A(0, 1) = p1[0] * P1(2, 1) - P1(0, 1);
    A(0, 2) = p1[0] * P1(2, 2) - P1(0, 2);
    A(0, 3) = p1[0] * P1(2, 3) - P1(0, 3); //A1

    A(1, 0) = p1[1] * P1(2, 0) - P1(1, 0);
    A(1, 1) = p1[1] * P1(2, 1) - P1(1, 1);
    A(1, 2) = p1[1] * P1(2, 2) - P1(1, 2);
    A(1, 3) = p1[1] * P1(2, 3) - P1(1, 3); //A2

    A(2, 0) = p2[0] * P2(2, 0) - P2(0, 0);
    A(2, 1) = p2[0] * P2(2, 1) - P2(0, 1);
    A(2, 2) = p2[0] * P2(2, 2) - P2(0, 2);
    A(2, 3) = p2[0] * P2(2, 3) - P2(0, 3); //A3

    A(3, 0) = p2[1] * P2(2, 0) - P2(1, 0);
    A(3, 1) = p2[1] * P2(2, 1) - P2(1, 1);
    A(3, 2) = p2[1] * P2(2, 2) - P2(1, 2);
    A(3, 3) = p2[1] * P2(2, 3) - P2(1, 3); //A4

    math::Matrix<double, 4, 4> V;
    math::matrix_svd<double, 4, 4>(A, nullptr, nullptr, &V);
    math::Vec3f X;
    X[0] = V(0, 3) / V(3, 3);
    X[1] = V(1, 3) / V(3, 3);
    X[2] = V(2, 3) / V(3, 3);

    std::cout << " trianglede point is :" << X[0] << " " << X[1] << " " << X[2] << std::endl;
    std::cout << " the result should be "
              << "2.14598 -0.250569 6.92321\n"
              << std::endl;

    return 0;
}
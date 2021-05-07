//给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

package Leetcode;

public class Leetcode_53 {
    public int maxSubArray(int[] nums) {
        int tmpSum = 0;
        int res = nums[0];

        for (int num : nums){
            tmpSum = Math.max(tmpSum + num,num);
            res = Math.max(res,tmpSum);
        }
        return res;
    }

    public static void main(String[] args) {

    }
}

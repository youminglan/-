package Leetcode;

import Leetcode.TreeNode;

//给定一个二叉树，检查它是否是镜像对称的。

public class Leetcode_06 {
    public boolean isSymmetric(TreeNode root) {
        return root == null ? true : recur(root.left,root.right);
    }
    boolean recur(TreeNode L,TreeNode R){
        if (L == null && R == null)
            return true;
        if (L == null || R == null || L.val != R.val)
            return false;
        return recur(L.left,R.right) && recur(L.right,R.left);
    }

    public static void main(String[] args) {

    }
}

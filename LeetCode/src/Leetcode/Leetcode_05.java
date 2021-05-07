package Leetcode;
import Leetcode.TreeNode;
//给定一个二叉树，判断其是否是一个有效的二叉搜索树。
//
//        假设一个二叉搜索树具有如下特征：
//
//        节点的左子树只包含小于当前节点的数。
//        节点的右子树只包含大于当前节点的数。
//        所有左子树和右子树自身必须也是二叉搜索树。

public class Leetcode_05 {
    long pre = Long.MIN_VALUE;
    public boolean isValidBST(TreeNode root) {
        if (root == null)
            return true;
        if(!isValidBST(root.left)){
            return false;
        }
        if(root.val<=pre){
            return false;
        }
        // 访问当前节点：如果当前节点小于等于中序遍历的前一个节点，说明不满足BST，返回 false；否则继续遍历。
        pre = root.val;
        return isValidBST(root.right);
    }

    public static void main(String[] args) {

    }
}

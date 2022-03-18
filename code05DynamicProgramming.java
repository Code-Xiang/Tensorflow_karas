public class code05{
    // 摆法符合题目意思：排完序之后，如果数字分布一样，就认为是同一种摆法！
    // 给你apples，给你plates
    // 返回几种摆法
    // 1000个
    // 1000个
    // 1000*1000
    public static int test(int apples, int plates) {
        // 1000
        // 1000
        int[][] dp = new int[apples+1][plates+1];
        for(int i = 0;i <= apples;i++){
            for(int j = 0;j <= plates;j++){
                dp[i][j] = -1;
            }
        }
        return f(apples, plates, dp);
    }
     public static int f(int apples, int plates, int[][] dp){
           if(dp[apples][plates] != -1) {
               return dp[apples][plates];
           }
           int ans = 0;
           if(apples == 0){
               ans = 1;
           }else if(plates == 0){
               ans = 0;
           }else{
            ans = f(apples-plates,plates, dp) + f(apples,plates-1, dp);
           }
           dp[apples][plates] = ans;
           return ans;
     }
}
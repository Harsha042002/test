const BASE_URL_CUSTOMER = 'http://localhost:8000';

interface LoginResponse {
  token: string;
  user: {
    id: string;
    name: string;
    mobile: string;
  };
}

export const authService = {
  // Send OTP
  async sendOTP(mobile: string): Promise<{ success: boolean; message: string }> {
    try {
      console.log('Sending OTP request to:', `${BASE_URL_CUSTOMER}/auth/sendotp`);
      console.log('Request payload:', { mobile });

      const response = await fetch(`${BASE_URL_CUSTOMER}/auth/sendotp`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ mobile }),
        credentials: 'include', // Include cookies for session management
      });

      console.log('Response status:', response.status);

      if (!response.ok) {
        const errorData = await response.json();
        console.error('Error response data:', errorData);
        throw new Error(errorData.message || 'Failed to send OTP');
      }

      console.log('OTP sent successfully');
      return { success: true, message: 'OTP sent successfully' };
    } catch (error: any) {
      console.error('Error in sendOTP:', error.message);
      throw new Error(error.message || 'Failed to send OTP');
    }
  },

  // Verify OTP
  async verifyOTP(mobile: string, otp: string): Promise<LoginResponse & { profile?: any }> {
    try {
      const response = await fetch(`${BASE_URL_CUSTOMER}/auth/verifyotp`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mobile, otp: parseInt(otp, 10), deviceId: 'web' }),
        credentials: 'include',
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.message || 'Failed to verify OTP');
      if (!data.token) throw new Error('Invalid response from server: Missing token');

      // Fetch profile to get canonical user info
      let profile = null;
      try {
        profile = await authService.getProfile();
      } catch (profileError) {
        console.error('Error fetching profile after login:', profileError);
      }

      // Always store user as { id, name, mobile }
      const userObj: { id: string | number; mobile: string; name?: string } = {
        id: (profile && profile.id) || data.user?.id,
        mobile:
          (profile && (profile.mobile || profile.phone)) ||
          data.user?.mobile ||
          data.user?.phone ||
          '',
      };
      const resolvedName =
        (profile && profile.name) ||
        data.user?.name ||
        undefined;
      if (resolvedName) userObj.name = resolvedName;

      localStorage.setItem('access_token', data.token);
      localStorage.setItem('user', JSON.stringify(userObj));

      return { ...data, profile: userObj };
    } catch (error: any) {
      console.error('Error in verifyOTP:', error.message);
      throw new Error(error.message || 'Failed to verify OTP');
    }
  },

  // Resend OTP
  async resendOTP(mobile: string): Promise<{ success: boolean; message: string }> {
    try {
      const response = await fetch(`${BASE_URL_CUSTOMER}/auth/resendotp`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ mobile }),
        credentials: 'include', // Include cookies for session management
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to resend OTP');
      }

      return { success: true, message: 'OTP resent successfully' };
    } catch (error: any) {
      console.error('Error in resendOTP:', error.message);
      throw new Error(error.message || 'Failed to resend OTP');
    }
  },

  // Logout
  async logout(): Promise<void> {
    try {
      const response = await fetch(`${BASE_URL_CUSTOMER}/auth/logout`, {
        method: 'GET',
        credentials: 'include',
      });
      
      if (!response.ok) {
        throw new Error('Logout failed');
      }
      
      // Clear all auth-related items from localStorage
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
      localStorage.removeItem('user');
      localStorage.removeItem('sessionId'); // Also remove session ID
      
    } catch (error) {
      console.error('Error in logout:', (error as any).message);
    } finally {
      // Ensure data is cleared even if request fails
      this.clearAuth();
    }
  },

  // Get Profile
  async getProfile(): Promise<any> {
    try {
      const response = await fetch(`${BASE_URL_CUSTOMER}/auth/profile`, {
        method: 'GET',
        credentials: 'include', // Include cookies for session management
      });

      if (!response.ok) {
        throw new Error('Failed to get profile');
      }

      return await response.json();
    } catch (error: any) {
      console.error('Error in getProfile:', error.message);
      throw new Error(error.message || 'Failed to get profile');
    }
  },

  // Clear Authentication Data
  clearAuth() {
    localStorage.removeItem('access_token');
    localStorage.removeItem('user');
  },

  // Get Token from LocalStorage
  getToken(): string | null {
    return localStorage.getItem('access_token');
  },

  // Get User from LocalStorage
  getUser() {
    const userStr = localStorage.getItem('user');
    return userStr ? JSON.parse(userStr) : null;
  },

  // Generic Fetch with Retry on 401
  async fetchWithRefresh(url: string, opts: RequestInit = {}): Promise<Response> {
    let response = await fetch(url, { ...opts, credentials: 'include' });

    if (response.status === 401) {
      // Refresh token and retry
      const refreshed = await this.refreshToken();
      if (!refreshed) {
        alert('Session expired—please login again');
        window.location.reload();
        return Promise.reject(new Error('Session expired'));
      }

      // Retry the original request
      response = await fetch(url, { ...opts, credentials: 'include' });
    }

    return response;
  },

  // Refresh Token
  async refreshToken(): Promise<boolean> {
    try {
      const response = await fetch(`${BASE_URL_CUSTOMER}/auth/refresh-token`, {
        method: 'GET',
        credentials: 'include', // Include cookies for session management
      });

      if (!response.ok) {
        return false;
      }

      const data = await response.json();
      localStorage.setItem('access_token', data.token);
      return true;
    } catch (error) {
      console.error('Error in refreshToken:', error);
      return false;
    }
  },
};
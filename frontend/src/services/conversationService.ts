// conversationService.ts (create this file in your services folder)
const BASE_URL = 'http://localhost:8000';

interface UserChat {
    conversation_id: string;
    timestamp: string;
    user_query: string;
    message_count: number;
}

interface ConversationDetail {
    conversation_id: string;
    session_id: string;
    timestamp: string;
    messages: Array<{
        role: string;
        content: string;
    }>;
}

export const conversationService = {
    async getUserConversations(): Promise<UserChat[]> {
        const token = localStorage.getItem('auth_token');
        if (!token) {
            throw new Error('Not authenticated');
        }

        const response = await fetch(`${BASE_URL}/user/conversations`, {
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`Failed to fetch: ${response.status}`);
        }

        return await response.json();
    },

    async getConversation(conversationId: string): Promise<ConversationDetail> {
        const token = localStorage.getItem('auth_token');
        if (!token) {
            throw new Error('Not authenticated');
        }

        const response = await fetch(`${BASE_URL}/conversations/${conversationId}`, {
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`Failed to fetch: ${response.status}`);
        }

        return await response.json();
    },

    async deleteAllUserConversations(): Promise<{ status: string, message: string }> {
        const token = localStorage.getItem('auth_token');
        if (!token) {
            throw new Error('Not authenticated');
        }

        const response = await fetch(`${BASE_URL}/user/conversations`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`Failed to delete: ${response.status}`);
        }

        return await response.json();
    }
};